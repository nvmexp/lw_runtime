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
#include <iostream>
#include <sstream>

#include "fm_log.h"
#include "FMCommonTypes.h"
#include "GFMHelper.h"
#include "FMDeviceProperty.h"
#include "GlobalFabricManager.h"
#include "FMAutoLock.h"

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#include "GlobalFmMulticastMgr.h"

GlobalFmMulticastMgr::GlobalFmMulticastMgr(GlobalFabricManager *pGfm)
{
    mpGfm = pGfm;

    lwosInitializeCriticalSection(&mLock);

    mMaxNumMcGroups = FMDeviceProperty::getMulticastRemapTableSize(mpGfm->getSwitchArchType());
    pMcGroupAllocated = NULL;
    mEnableSpray = true;
    mNoDynRsp = false;

    // initially set all the supported group's allocated state as false
    if ( mMaxNumMcGroups > 0 ) {
        pMcGroupAllocated = new bool[mMaxNumMcGroups];
        memset(pMcGroupAllocated, 0, sizeof(bool)*mMaxNumMcGroups);
    }

    mGroupTable.clear();
}

GlobalFmMulticastMgr::~GlobalFmMulticastMgr()
{
    lwosDeleteCriticalSection(&mLock);

    if (pMcGroupAllocated) {
        delete [] pMcGroupAllocated;
    }

    MulticastGroupTable::iterator it;
    for ( it = mGroupTable.begin(); it != mGroupTable.end(); it++ ) {
        MulticastGroupInfo *pGroupInfo = it->second;
        if (pGroupInfo) {
            delete pGroupInfo;
        }
    }
    mGroupTable.clear();

}

//
// allocate multicast group Id on the specified partition
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//
// return
//   FM_INT_ST_OK, if a free multicast group is found
//   FM_INT_ST_RESOURCE_UNAVAILABLE, if there is no free multicast group
//
FMIntReturn_t
GlobalFmMulticastMgr::allocateMulticastGroup(uint32_t partitionId, uint32_t &groupId)
{
    uint32_t nodeId = 0;

    // Serialize multicast APIs
    FMAutoLock lock(mLock);

    if  ( ( partitionId != ILWALID_FABRIC_PARTITION_ID ) &&
          ( !mpGfm->mGfmPartitionMgr->isPartitionActive(nodeId, partitionId) ) ) {
        FM_LOG_ERROR("multicast group allocation requested for non active partition id %d.", partitionId);
        return FM_INT_ST_BADPARAM;
    }

    MulticastGroupInfo * pGroupInfo = getFreeMulticastGroup(partitionId);

    if ( pGroupInfo == NULL ) {
        // no unused group found
        return FM_INT_ST_RESOURCE_UNAVAILABLE;
    }

    groupId = pGroupInfo->mcId;

    if ( partitionId == ILWALID_FABRIC_PARTITION_ID ) {
        FM_LOG_INFO("multicast group %d is allocated.", groupId);
    } else {
        FM_LOG_INFO("multicast group %d for partition id %d is allocated.", groupId, partitionId);
    }

    return FM_INT_ST_OK;
}

//
// free specified multicast group Id on a partition
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//
FMIntReturn_t
GlobalFmMulticastMgr::freeMulticastGroup(uint32_t partitionId, uint32_t groupId)
{
    // Serialize multicast APIs
    FMAutoLock lock(mLock);

    MulticastGroupTable::iterator it;
    it = getMulticastGroupIterator(partitionId, groupId);
    if ( it == mGroupTable.end() ) {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_LOG_ERROR("the specified multicast group id %d for partition id %d is not allocated.",
                         groupId, partitionId);
        } else {
            FM_LOG_ERROR("the specified multicast group id %d is not allocated.", groupId);
        }
        return FM_INT_ST_BADPARAM;
    }

    MulticastGroupInfo *pGroupInfo = it->second;
    mGroupTable.erase(it);

    // free all resources used in this group, and free the group
    freeMulticastGroupInfo(pGroupInfo);

    return FM_INT_ST_OK;
}

//
// free all multicast group Ids on a specified partition
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//
void
GlobalFmMulticastMgr::freeMulticastGroups(uint32_t partitionId)
{
    // Serialize multicast APIs
    FMAutoLock lock(mLock);

    MulticastGroupTable::iterator it = mGroupTable.begin();

    while ( it != mGroupTable.end() ) {

        MulticastGroupKeyType key = it->first;

        // keep the current iterator and then increase it
        MulticastGroupTable::iterator lwrr = it++;

        if ( key.partitionId == partitionId ) {

            MulticastGroupInfo *pGroupInfo = lwrr->second;
            mGroupTable.erase(lwrr);

            // free all resources used in this group, and free the group
            freeMulticastGroupInfo(pGroupInfo);
        }
    }
}

//
// get available multicast group Ids on a specified partition
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//
void
GlobalFmMulticastMgr::getAvailableMulticastGroups(uint32_t partitionId, std::list<uint32_t> &groupIds)
{
    // Serialize multicast APIs
    FMAutoLock lock(mLock);

    groupIds.clear();

    for ( uint32_t i = 0; i < mMaxNumMcGroups; i++ ) {
        if ( pMcGroupAllocated[i] == false ) {

            // found an unused group
            groupIds.push_back(i);
        }
    }
}

//
// set multicast group config on a already allocated group
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//  reflectiveMode: set to true if the group is in reflective memory mode
//  excludeSelf:    set to true if the multicast port list should exclude requesting GPU itself
//  gpus:           set of GPUs that are participating in this multicast group
//  primaryReplica: the primaryReplica GPU in this multicast group
//
FMIntReturn_t
GlobalFmMulticastMgr::setMulticastGroup(uint32_t partitionId, uint32_t groupId,
                                        bool reflectiveMode, bool excludeSelf,
                                        std::set<GpuKeyType> &gpus, GpuKeyType primaryReplica)
{
    // Serialize multicast APIs
    FMAutoLock lock(mLock);

    MulticastGroupKeyType key;
    key.partitionId = partitionId;
    key.groupId = groupId;

    MulticastGroupTable::iterator it = mGroupTable.find(key);
    if ( it == mGroupTable.end() ) {

        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_LOG_ERROR("failed to find multicast group %d for partition id %d. The group may not be allocated for the specified partition.",
                         groupId, partitionId);
        } else {
            FM_LOG_ERROR("failed to find multicast group %d. The group may not be allocated.", groupId);
        }
        return FM_INT_ST_BADPARAM;
    }

    if ( gpus.size() == 0 ) {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_LOG_ERROR("multicast join requested with empty set of GPUs for group %d for partition id %d.",
                         groupId, partitionId);
        } else {
            FM_LOG_ERROR("multicast join requested with empty set of GPUs for group %d.", groupId);
        }
        return FM_INT_ST_BADPARAM;
    }

    // check if the GPUs are in the specified partition
    if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
        for ( std::set<GpuKeyType>::iterator it = gpus.begin(); it != gpus.end(); it++ ) {
            GpuKeyType key = *it;

            if ( !mpGfm->mGfmPartitionMgr->isGpuUsedInPartition(partitionId, key) ) {
                FM_LOG_ERROR("failed to join multicast group as the requested GPU physical id %d is not part of the specified partition id %d.",
                             key.physicalId, partitionId);
                return FM_INT_ST_BADPARAM;
            }
        }
    }

    std::set<GpuKeyType>::iterator jit;
    jit = gpus.find(primaryReplica);
    if ( jit == gpus.end() ) {
        FM_LOG_ERROR("Primary Replica GPU has not joined in the multicast group.");
        return FM_INT_ST_BADPARAM;
    }

    MulticastGroupInfo *pGroupInfo = it->second;
    pGroupInfo->partitionId = partitionId;
    pGroupInfo->mcId = groupId;
    pGroupInfo->reflectiveMode = reflectiveMode;
    pGroupInfo->excludeSelf = excludeSelf;
    pGroupInfo->noDynRsp = mNoDynRsp;
    pGroupInfo->gpuList = gpus;
    pGroupInfo->primaryReplica = primaryReplica;

    std::ostringstream ss;
    ss << "adding " <<  pGroupInfo->gpuList.size() << " GPU ";
    ;
    for ( jit = pGroupInfo->gpuList.begin(); jit != pGroupInfo->gpuList.end(); jit++ ) {

        GpuKeyType key = *jit;
        ss << key.nodeId << "/" << key.physicalId << " ";
    }

    if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
        ss << "to partition " << pGroupInfo->partitionId << " multicast group " << pGroupInfo->mcId;
    } else {
        ss << "to multicast group " << pGroupInfo->mcId;
    }
    ss << " with Primary Replica as " << pGroupInfo->primaryReplica.nodeId
            << "/" << pGroupInfo->primaryReplica.physicalId << std::endl;
    FM_LOG_INFO("%s", ss.str().c_str());

    FMIntReturn_t rc;
    // construct multicast tree
    rc = constructMulticastTree(pGroupInfo);
    if ( rc != FM_INT_ST_OK ) {
        handleMulticastConfigError(pGroupInfo);
        return rc;
    }

    // send the switch routing to LFM
    rc = configMulticastRoutes(pGroupInfo, false);
    if ( rc != FM_INT_ST_OK ) {
        handleMulticastConfigError(pGroupInfo);
        return rc;
    }

    return FM_INT_ST_OK;
}

//
// set multicast group config on a already allocated group, the first GPU in the GPU
// list is selected as primaryReplica GPU in this multicast group
// use ILWALID_FABRIC_PARTITION_ID for bare metal
//
//  reflectiveMode: set to true if the group is in reflective memory mode
//  excludeSelf:    set to true if the multicast port list should exclude requesting GPU ifself
//  gpus:           set of GPUs that are participating in this multicast group
//
FMIntReturn_t
GlobalFmMulticastMgr::setMulticastGroup(uint32_t partitionId, uint32_t groupId,
                                        bool reflectiveMode, bool excludeSelf,
                                        std::set<GpuKeyType> &gpus)
{
    // Serialize multicast APIs
    // Lock will be taken at setMulticastGroup

    if ( gpus.size() == 0 ) {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_LOG_ERROR("no GPUs in multicast group %d for partition id %d.",
                         groupId, partitionId);
        } else {
            FM_LOG_ERROR("no GPUs in multicast group %d.", groupId);
        }
        return FM_INT_ST_BADPARAM;
    }

    // no primaryReplica is specified, use the first GPU in the list
    std::set<GpuKeyType>::iterator it = gpus.begin();
    GpuKeyType primaryReplica = *it;

    return setMulticastGroup(partitionId, groupId, reflectiveMode, excludeSelf,
                             gpus, primaryReplica);
}

//
// return the multicast base address for a specified multicast group Id
// This is a helper function, no need to take lock.
//
FMIntReturn_t
GlobalFmMulticastMgr::getMulticastGroupBaseAddress(uint32_t groupId, uint64_t &multicastAddrBase)
{
    if ( groupId >= mMaxNumMcGroups ) {
        return FM_INT_ST_BADPARAM;
    }

    multicastAddrBase = FMDeviceProperty::getMulticastBaseAddrFromGroupId(mpGfm->getSwitchArchType(), groupId);
    return FM_INT_ST_OK;
}

//
// return the multicast base address and Range for a specified multicast group Id
// This is a helper function, no need to take lock.
//
FMIntReturn_t
GlobalFmMulticastMgr::getMulticastGroupBaseAddrAndRange(uint32_t groupId, uint64_t &multicastAddrBase,
                                                        uint64_t &multicastAddrRange)
{
    if ( groupId >= mMaxNumMcGroups ) {
        return FM_INT_ST_BADPARAM;
    }

    multicastAddrBase = FMDeviceProperty::getMulticastBaseAddrFromGroupId(mpGfm->getSwitchArchType(), groupId);
    multicastAddrRange = FMDeviceProperty::getAddressRangePerGpu(mpGfm->getSwitchArchType());
    return FM_INT_ST_OK;
}

// print the internal data of the specified multicast group.
void
GlobalFmMulticastMgr::dumpMulticastGroup( uint32_t partitionId, uint32_t groupId, std::stringstream &outStr )
{
    MulticastGroupInfo *pGroupInfo = NULL;

    MulticastGroupKeyType key;
    key.partitionId = partitionId;
    key.groupId = groupId;

    MulticastGroupTable::iterator it = mGroupTable.find(key);

    if ( it != mGroupTable.end() ) {
        pGroupInfo = it->second;
    }

    if ( !pGroupInfo ) {
        outStr << "Multicast group "<< groupId << " for partition " << partitionId << " is not used. " << endl;
        std::string strInfo = outStr.str();
        FM_LOG_INFO("%s", strInfo.c_str());
        return;
    }

    dumpMulticastGroupInfo(pGroupInfo, outStr);
}

void
GlobalFmMulticastMgr::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    switch (pFmMessage->type())
    {
        case lwswitch::FM_MULTICAST_GROUP_CREATE_REQ:
        {
            handleGroupCreateReqMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_BIND_REQ:
        {
            handleGroupBindReqMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_ACK:
        {
            handleGroupSetupCompleteAckMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_REQ:
        {
            handleGroupReleaseReqMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_COMPLETE_ACK:
        {
            handleGroupReleaseCompleteAckMsg(pFmMessage);
            break;
        }
        default:
        {
            FM_LOG_ERROR("unknown multicast message type %d from " NODE_ID_LOG_STR " %d",
                         pFmMessage->type(), pFmMessage->nodeid());
            break;
        }
    }
}

void
GlobalFmMulticastMgr::dumpMcIdEntry(PortMcIdTableEntry &entry, std::stringstream &outStr)
{


    outStr << "port: " << entry.portNum << " extendedTable: " << entry.extendedTable << " index: "
             << entry.index << " extendedTablePtrValid: " << entry.extendedTablePtrValid << endl;

    outStr << "      Sprays: " << endl;

    MulticastSprays::iterator sprayIt;

    for ( sprayIt = entry.sprays.begin(); sprayIt != entry.sprays.end(); sprayIt++ ) {
        MulticastPortList &spray = *sprayIt;
        outStr << "      ";

        MulticastPorts::iterator portIt;
        for ( portIt = spray.ports.begin(); portIt != spray.ports.end(); portIt++ ) {
            MulticastPort &port = *portIt;
            outStr << "(" << port.portNum << "," << port.vcHop << ") ";
        }

        outStr << " replicaValid: " << spray.replicaValid;
        outStr << " replicaOffset: " << spray.replicaOffset << endl;
    }
}

void
GlobalFmMulticastMgr::dumpMulticastGroupInfo(MulticastGroupInfo *groupInfo, std::stringstream &outStr)
{
    if ( !groupInfo ) {
        return;
    }

    outStr << "mcId:           " << groupInfo->mcId << endl;
    outStr << "partitionId:    " << groupInfo->partitionId << endl;
    outStr << "reflectiveMode: " << groupInfo->reflectiveMode << endl;
    outStr << "excludeSelf:    " << groupInfo->excludeSelf << endl;

    outStr << "GPUs:           " << endl;
    std::set<GpuKeyType>::iterator it;
    for ( it = groupInfo->gpuList.begin(); it != groupInfo->gpuList.end(); it++ ) {
        GpuKeyType gpuKey = *it;
        outStr << gpuKey.nodeId << "/" << gpuKey.physicalId << " ";
    }
    outStr << endl;

    outStr << "primaryReplica: " << groupInfo->primaryReplica.nodeId << "/"
            << groupInfo->primaryReplica.physicalId << endl;
    outStr << endl;

    outStr << "Multicast Table: " << endl;
    NodeMulticastTable::iterator nodeIt;
    for ( nodeIt = groupInfo->multicastTable.begin(); nodeIt != groupInfo->multicastTable.end(); nodeIt++ ) {

        const uint32_t nodeId = nodeIt->first;
        outStr << NODE_ID_LOG_STR << nodeId << endl;

        SwitchMulticastTable &switchMcastTbl = nodeIt->second;
        SwitchMulticastTable::iterator switchTblIt;

        for ( switchTblIt = switchMcastTbl.begin(); switchTblIt != switchMcastTbl.end(); switchTblIt++ ) {

            const SwitchKeyType &switchKey = switchTblIt->first;

            outStr << "switch: " << switchKey.nodeId << "/" << switchKey.physicalId << endl;

            PortMulticastTable &portMcastTbl = switchTblIt->second;
            PortMulticastTable::iterator portTblIt;

            for ( portTblIt = portMcastTbl.begin(); portTblIt != portMcastTbl.end(); portTblIt++ ) {

                const PortKeyType portKey = portTblIt->first;
                outStr << "port: " << portKey.nodeId << "/" << portKey.physicalId
                        << "/" << portKey.portIndex << endl;

                PortMulticastInfo &portMcastInfo = portTblIt->second;

                PortMcIdTableEntry &mcIdEntry = portMcastInfo.mcIdEntry;
                outStr << "port " << portMcastInfo.portNum << " mcId Table: " << endl;
                dumpMcIdEntry(mcIdEntry, outStr);

                PortMcIdTableEntry &extendedcIdTable = portMcastInfo.extendedMcIdEntry;
                outStr << "port " << portMcastInfo.portNum << " extended mcId Table: " << endl;
                dumpMcIdEntry(extendedcIdTable, outStr);

                outStr << endl;
            }
            outStr << endl;
        }
        outStr << endl;
    }

    outStr << std::endl;
    std::string strInfo = outStr.str();
    FM_LOG_INFO("%s", strInfo.c_str());

}

FMIntReturn_t
GlobalFmMulticastMgr::constructMulticastTree(MulticastGroupInfo *groupInfo)
{
    if ( !groupInfo ) {
        return FM_INT_ST_BADPARAM;
    }

    //
    // get the latest active link mask on all switches, so that inactive links will not
    // be included in the multicast route
    //
    std::map<SwitchKeyType, uint64_t> switchActiveLinkMask;
    std::map<SwitchKeyType, uint64_t>::iterator switchIt;
    getAllSwitchActiveLinkMask( groupInfo->partitionId, switchActiveLinkMask );

    // program multicast table on active links
    for (switchIt = switchActiveLinkMask.begin(); switchIt != switchActiveLinkMask.end(); switchIt++) {
        SwitchKeyType switchKey = switchIt->first;
        uint64_t activeLinkMask = switchIt->second;

        uint32_t srcPortNum;
        for (srcPortNum = 0; srcPortNum < MAX_PORTS_PER_LWSWITCH; srcPortNum++) {
            if ( (activeLinkMask & (1LL << srcPortNum)) == 0 ) {
                // The port is either not active or not used in the partition

                // TODO: need to check
                // this might not be true if the multicast group is set before
                // the partition is activated, the links are not active yet.
                continue;
            }

            GpuTargetIds srcGpuSet, dstGpuSet;
            PortKeyType srcPortKey;
            srcPortKey.nodeId = switchKey.nodeId;
            srcPortKey.physicalId = switchKey.physicalId;
            srcPortKey.portIndex = srcPortNum;

            bool isSrcPortAccess = mpGfm->mpParser->isAccessPort( srcPortKey.nodeId,
                                                                  srcPortKey.physicalId,
                                                                  srcPortKey.portIndex );

            if (!sourcePortCheck(groupInfo, srcPortKey, isSrcPortAccess, srcGpuSet, dstGpuSet)) {
                // no need to program the source port for this multicast group
                // as this port is not ilwolved in this group
                continue;
            }

            // now find dstPorts to all GPUs in dstGpuSet from the srcPort
            uint64_t primaryReplicaDstPorts;  // destination port mask to the primary replica
            std::set<uint64_t> dstPortGroups; // destination port masks to other GPUs in the group

            if (!getDstPorts(groupInfo, switchKey, srcPortKey, activeLinkMask, dstGpuSet,
                             primaryReplicaDstPorts, dstPortGroups)) {
                continue;
            }

            // set the destination dstPorts found in the multicast table
            setPortMulticastTbl(groupInfo, switchKey, srcPortNum,
                                primaryReplicaDstPorts, dstPortGroups);
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFmMulticastMgr::configMulticastRoutes(MulticastGroupInfo *groupInfo, bool freeGroup)
{
    if ( !groupInfo ) {
        return FM_INT_ST_BADPARAM;
    }

    FMIntReturn_t rc;
    NodeMulticastTable::iterator it;

    for (it = groupInfo->multicastTable.begin(); it != groupInfo->multicastTable.end(); it++) {

        SwitchMulticastTable &switchMcastTable = it->second;
        SwitchMulticastTable::iterator jit;

        for ( jit = switchMcastTable.begin(); jit != switchMcastTable.end(); jit++ ) {

            const SwitchKeyType &switchKey = jit->first;
            PortMulticastTable &portMcastTable = jit->second;

            // get the mapped address of this group, mapped to it's parent's FLA address
            uint64_t mappedAddr, mappedAddrRange;
            rc = getMulticastGroupBaseAddrAndRange(groupInfo->parentMcId, mappedAddr, mappedAddrRange);
            if ( rc != FM_INT_ST_OK ){
                FM_LOG_ERROR("failed to map multicast address on " NODE_ID_LOG_STR " %d switch physical id %d",
                             switchKey.nodeId, switchKey.physicalId);
                return rc;
            }

            //
            // configure the multicast remap synchronously
            // If any error oclwrred during any of the following configuration, the error will
            // be returned to the API caller, FM and RM will coordinate in clean up the group.
            //
            rc = mpGfm->mpConfig->configMulticastRemapTable(switchKey.nodeId, switchKey.physicalId, groupInfo,
                                                            portMcastTable, mappedAddr, freeGroup, true);
             if ( rc != FM_INT_ST_OK ){
                 FM_LOG_ERROR("failed to configure multicast route on " NODE_ID_LOG_STR " %d switch physical id %d",
                              switchKey.nodeId, switchKey.physicalId);
                 return rc;
             }

            // configure the multicast RID synchronously
            rc = mpGfm->mpConfig->configMulticastRoutes(switchKey.nodeId, switchKey.physicalId, groupInfo,
                                                        portMcastTable, freeGroup, true);
            if ( rc != FM_INT_ST_OK ){
                FM_LOG_ERROR("failed to configure multicast route on " NODE_ID_LOG_STR " %d switch physical id %d",
                             switchKey.nodeId, switchKey.physicalId);
                return rc;
            }
        }
    }

    return FM_INT_ST_OK;
}

void
GlobalFmMulticastMgr::handleMulticastConfigError( MulticastGroupInfo *groupInfo )
{
    // TODO
}

//
// TODO:
// - add a config to allow or disallow multicast group share sharing among partitions
// - allocate group on the specified partition
//
MulticastGroupInfo *
GlobalFmMulticastMgr::getFreeMulticastGroup(uint32_t partitionId)
{
    MulticastGroupInfo * pGroupInfo = NULL;

    for ( uint32_t i = 0; i < mMaxNumMcGroups; i++ ) {
        if ( pMcGroupAllocated[i] == false ) {
            // found an unused group

            // create a new group
            MulticastGroupKeyType key;
            key.partitionId = partitionId;
            key.groupId = i;

            pGroupInfo = new MulticastGroupInfo;
            memset(pGroupInfo, 0, sizeof(MulticastGroupInfo));
            mGroupTable.insert(make_pair(key, pGroupInfo));

            pGroupInfo->partitionId = partitionId;
            pGroupInfo->mcId = i;
            pGroupInfo->parentMcId = i; // init the parent mcId the same as mcId
            pGroupInfo->gpuList.clear();
            pGroupInfo->multicastTable.clear();

            // Mark the group as allocated
            pMcGroupAllocated[i] = true;
            break;
        }
    }

    return pGroupInfo;
}

void
GlobalFmMulticastMgr::clearPortMcIdEntry( PortMcIdTableEntry &entry )
{
    MulticastSprays::iterator jit;
    for ( jit = entry.sprays.begin(); jit != entry.sprays.end(); jit++ ) {
        MulticastPortList &portList = *jit;

        // clear all ports
        portList.ports.clear();
    }

    // clear all spray groups
    entry.sprays.clear();
}

void
GlobalFmMulticastMgr::clearMulticastGroup( MulticastGroupInfo *groupInfo )
{
    if ( !groupInfo ) {
        return;
    }

    groupInfo->gpuList.clear();

    NodeMulticastTable::iterator nodeIt;
    for ( nodeIt = groupInfo->multicastTable.begin();
          nodeIt != groupInfo->multicastTable.end(); nodeIt++ ) {

        SwitchMulticastTable &switchTable = nodeIt->second;
        SwitchMulticastTable::iterator switchIt;

        for ( switchIt = switchTable.begin(); switchIt != switchTable.end(); switchIt++ ) {

            PortMulticastTable &portTable = switchIt->second;
            PortMulticastTable::iterator portIt;

            for ( portIt = portTable.begin(); portIt != portTable.end(); portIt++ ) {

                PortMulticastInfo &portInfo = portIt->second;
                clearPortMcIdEntry(portInfo.mcIdEntry);
                clearPortMcIdEntry(portInfo.extendedMcIdEntry);
            }
            portTable.clear();
        }
        switchTable.clear();
    }
    groupInfo->multicastTable.clear();
}

void
GlobalFmMulticastMgr::freeMulticastGroupInfo(MulticastGroupInfo *groupInfo)
{
    if (!groupInfo) {
        return;
    }

    uint32_t groupId = groupInfo->mcId;
    uint32_t partitionId = groupInfo->partitionId;

    // free switch routing table by setting all mcid entries for this group invalid
    configMulticastRoutes(groupInfo, true);

    // clear and free the group data
    clearMulticastGroup(groupInfo);
    delete groupInfo;

    // Mark the group as freed
    pMcGroupAllocated[groupId] = false;

    if ( partitionId == ILWALID_FABRIC_PARTITION_ID ) {
        FM_LOG_INFO("multicast group %d is freed.", groupId);
    } else {
        FM_LOG_INFO("multicast group %d for partition id %d is freed.", groupId, partitionId);
    }
}

void
GlobalFmMulticastMgr::getAllSwitchActiveLinkMask( uint32_t partitionId,
                                                  std::map<SwitchKeyType, uint64_t> &switchActiveLinkMask )
{
    switchActiveLinkMask.clear();

    FMLWSwitchInfoMap switchInfoMap = mpGfm->getLwSwitchInfoMap();
    FMLWSwitchInfoMap::iterator it;

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        uint32_t nodeId = it->first;

        FMLWSwitchInfoList switchList = it->second;
        FMLWSwitchInfoList::iterator jit;

        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            FMLWSwitchInfo switchInfo = (*jit);

            SwitchKeyType key;
            key.nodeId = nodeId;
            key.physicalId = switchInfo.physicalId;

            uint64_t activeLinkMask = switchInfo.enabledLinkMask;
            uint64_t partitionEnabledLinkMask = 0;
            PartitionSwitchInfo partitionSwitchInfo;

            if ( partitionId != ILWALID_FABRIC_PARTITION_ID )  {
                if  (!mpGfm->mGfmPartitionMgr->getSwitchEnabledLinkMaskForPartition(partitionId, key,
                                                                  partitionEnabledLinkMask) == false ) {
                    // the switch is not used in the partition
                    continue;
                } else {
                    activeLinkMask &= partitionEnabledLinkMask;
                }
            }

            // with ALI, get the LWLink mask that are trained to high speed
            // Laguna TODO
            activeLinkMask &= 0xFFFFFFFFFFFFFFFF;

            switchActiveLinkMask.insert(std::make_pair(key, activeLinkMask));
        }
    }
}

//
// if the port cannot reach any of the GPUs in the multicast group
//    there is no need to program
//    return false
//
// else
//    return true
//    find the GPUs srcGpuSet that
//      are directly connected to this source access port
//      or can be reached from this source trunk port
//    so that these GPUs can be removed them from the destination GPUs dstGpuSet
//    to prevent loop
//
bool
GlobalFmMulticastMgr::sourcePortCheck(MulticastGroupInfo *groupInfo, PortKeyType srcPortKey,
                                      bool isSrcPortAccess, GpuTargetIds &srcGpuSet,
                                      GpuTargetIds &dstGpuSet)
{
    bool rc = false;
    srcGpuSet.clear();
    dstGpuSet.clear();

    SwitchKeyType switchKey;
    switchKey.nodeId = srcPortKey.nodeId;
    switchKey.physicalId = srcPortKey.physicalId;

    GpusReachableFromPortMap::iterator it;
    it = mpGfm->mpParser->gpusReachableFromPortMap.find(switchKey);
    if ( it == mpGfm->mpParser->gpusReachableFromPortMap.end() ) {
        // No GPU reachable from this switch
        FM_LOG_DEBUG("%s: no GPU is reachable from switch %d/0x%x",
                    __FUNCTION__, switchKey.nodeId, switchKey.physicalId);
        return rc;
    }

    GpusReachableFromPort &gpusReachableFromPort = it->second;
    GpusReachableFromPort::iterator jit;
    jit = gpusReachableFromPort.find( srcPortKey.portIndex );
    if ( jit == gpusReachableFromPort.end() ) {
        // No GPU is reachable from this port
        FM_LOG_DEBUG("%s: no GPU is reachable from port %d/0x%x/%d",
                    __FUNCTION__, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
        return rc;
    }

    if ( areGPUsReachableOnPort(groupInfo, srcPortKey) == false ) {
        // no GPU from this multicast group is reachable from this port
        FM_LOG_DEBUG("%s: no GPU from this multicast group is reachable from port %d/0x%x/%d",
                    __FUNCTION__, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
        return rc;
    }

    // GPUs that are reachable from this port
    srcGpuSet = jit->second;

    // make sure at lease one of the GPUs in this multicast group
    // is reachable from this port, if not this port is not relevant to
    // this multicast group.
    std::set<GpuKeyType>::iterator dstGpuKeyIt;

    for ( dstGpuKeyIt = groupInfo->gpuList.begin(); dstGpuKeyIt != groupInfo->gpuList.end(); dstGpuKeyIt++ ) {
        GpuKeyType gpuKey = *dstGpuKeyIt;
        uint32_t targetId;

        if ( !mpGfm->mpParser->getGpuTargetIdFromKey( gpuKey, targetId ) ) {
            FM_LOG_ERROR("Failed to find targetId for GPU %d/%d.",
                         gpuKey.nodeId, gpuKey.physicalId);
            continue;
        }

        if ( srcGpuSet.find(targetId) != srcGpuSet.end() ) {

            // The GPU can be reached from the source port
            FM_LOG_DEBUG("%s: GPU %d is reachable from source port %d/0x%x/%d.",
                        __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
            rc = true;

            if ( isSrcPortAccess ) {
                // do not remove directly connected GPU to this source port
                // because the group is set to be in reflectiveMode
                if ( groupInfo->reflectiveMode) {
                    dstGpuSet.insert(targetId);
                    FM_LOG_DEBUG("%s: The group is in reflective mode, do not remove GPU %d from source port %d/0x%x/%d.",
                                __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
                    FM_LOG_DEBUG("%s: Add GPU %d to source port %d/0x%x/%d to destination GPU list.",
                                __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
                    continue;
                }

                // do not remove directly connected GPU to this source port
                // because the group is set to include source GPU itseself
                if ( groupInfo->reflectiveMode || !groupInfo->excludeSelf ) {
                    dstGpuSet.insert(targetId);
                    FM_LOG_DEBUG("%s: The group is not set to exclude self, do not remove GPU %d from source port %d/0x%x/%d.",
                                __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
                    FM_LOG_DEBUG("%s: Add GPU %d to source port %d/0x%x/%d to destination GPU list.",
                                __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
                    continue;
                }
            }

            // The GPU can be reached from the source port
            // do not include the GPU in the destination list to
            // prevent loop
            FM_LOG_DEBUG("%s: Remove GPU %d from source port %d/0x%x/%d to prevent loop.",
                        __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
            continue;
        }

        // The GPU can not be reached from the source port
        FM_LOG_DEBUG("%s: Add GPU %d to source port %d/0x%x/%d to destination GPU list.",
                    __FUNCTION__, targetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
        dstGpuSet.insert(targetId);
        rc = true;
    }
    return rc;
}

// get destination ports to the primary replica and other GPUs in the group
//   primaryReplicaDstPorts port mask to the primary replica
//   dstPorts to all other GPUs in dstGpuSet
bool
GlobalFmMulticastMgr::getDstPorts( MulticastGroupInfo *groupInfo, SwitchKeyType switchKey,
                                   PortKeyType srcPortKey, uint64_t activeLinkMask, GpuTargetIds &dstGpuSet,
                                   uint64_t &primaryReplicaDstPorts, std::set<uint64_t> &dstPortGroups )
{
    primaryReplicaDstPorts = 0;
    dstPortGroups.clear();

    PortsToReachGpuMap::iterator switchMapIt = mpGfm->mpParser->portsToReachGpuMap.find(switchKey);
    if ( switchMapIt == mpGfm->mpParser->portsToReachGpuMap.end() ) {
        FM_LOG_ERROR("No port can reach GPUs on the switch " NODE_ID_LOG_STR " %d, physical id %d.",
                     switchKey.nodeId, switchKey.physicalId);
        return false;
    }

    uint32_t primaryReplicaTargetId;
    if ( !mpGfm->mpParser->getGpuTargetIdFromKey( groupInfo->primaryReplica, primaryReplicaTargetId ) ) {
        FM_LOG_ERROR("Failed to get GPU " NODE_ID_LOG_STR " %d, physical id %d targetId.",
                     groupInfo->primaryReplica.nodeId, groupInfo->primaryReplica.physicalId);
        return false;
    } else {
        FM_LOG_DEBUG("%s: primaryReplicaTargetId %d.", __FUNCTION__, primaryReplicaTargetId);
    }

    // first find the port mask to reach primary replica
    if ( dstGpuSet.find(primaryReplicaTargetId) != dstGpuSet.end() ) {

        getDstPortMaskToGpu(srcPortKey, primaryReplicaTargetId, primaryReplicaDstPorts);
        primaryReplicaDstPorts &= activeLinkMask;

        if ( primaryReplicaDstPorts == 0 ){
            FM_LOG_ERROR("No active port can reach Primary Replica GPU %d on switch " NODE_ID_LOG_STR " %d, physical id %d, port %d.",
                         primaryReplicaTargetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
            return false;
        }

        FM_LOG_DEBUG("%s: primaryReplicaTargetId %d, primaryReplicaDstPorts 0x%lx",
                    __FUNCTION__, primaryReplicaTargetId, primaryReplicaDstPorts);
    }

    // find port masks to reach GPUs except primary replica
    for ( GpuTargetIds::iterator dstGpuIt = dstGpuSet.begin(); dstGpuIt != dstGpuSet.end(); dstGpuIt++ ) {
        uint32_t dstTargetId = *dstGpuIt;

        if ( dstTargetId == primaryReplicaTargetId ) {
            continue;
        }

        uint64_t dstPortMask;
        getDstPortMaskToGpu(srcPortKey, dstTargetId, dstPortMask);
        dstPortMask &= activeLinkMask;

        if ( dstPortMask == 0 ){
            FM_LOG_ERROR("No active port can reach GPU %d on switch " NODE_ID_LOG_STR " %d, physical id %d, port %d.",
                         dstTargetId, srcPortKey.nodeId, srcPortKey.physicalId, srcPortKey.portIndex);
            return false;
        }

        // remove any ports that are already in primaryReplicaDstPorts
        dstPortMask &= ~primaryReplicaDstPorts;
        if (dstPortMask) {
            dstPortGroups.insert(dstPortMask);
        }
        FM_LOG_DEBUG("%s: dstTargetId %d, dstPortMask 0x%lx",
                     __FUNCTION__, dstTargetId, dstPortMask);
    }

    if (dstPortGroups.size() == 0 ) {
        FM_LOG_DEBUG("Failed to find any destination port to partition %d"
                     "multicast group %d on switch " NODE_ID_LOG_STR " %d, physical id %d.",
                     groupInfo->partitionId, groupInfo->mcId,
                     switchKey.nodeId, switchKey.physicalId);
        return false;
    }

    return true;
}

void
GlobalFmMulticastMgr::portMaskTolist(uint64_t portMask, std::vector<uint32_t> &portList)
{
    uint32_t i = 0;
    portList.clear();

    while (portMask) {
        if (portMask & 1LL) {
            portList.push_back(i);
        }

        i++;
        portMask >>= 1;
    }
}

void
GlobalFmMulticastMgr::setPortMulticastTbl(MulticastGroupInfo *groupInfo, SwitchKeyType switchKey, uint32_t srcPortNum,
                                          uint64_t &primaryReplicaDstPorts, std::set<uint64_t> &dstPortGroups)
{
    FM_LOG_DEBUG("%s: mcId %d, source port %d/0x%x/%d, primaryReplicaDstPorts 0x%016lx",
                __FUNCTION__, groupInfo->mcId, switchKey.nodeId, switchKey.physicalId,
                srcPortNum, primaryReplicaDstPorts);

    NodeMulticastTable &nodeTbl = groupInfo->multicastTable;
    NodeMulticastTable::iterator nodeIt = nodeTbl.find(switchKey.nodeId);

    if (nodeIt == nodeTbl.end()) {
        SwitchMulticastTable switchMulticastTable;
        switchMulticastTable.clear();
        nodeTbl.insert(make_pair(switchKey.nodeId, switchMulticastTable));
    }

    nodeIt = nodeTbl.find(switchKey.nodeId);
    SwitchMulticastTable &switchTbl = nodeIt->second;

    SwitchMulticastTable::iterator switchIt = switchTbl.find(switchKey);

    if (switchIt == switchTbl.end()) {
        PortMulticastTable portMulticastTable;
        portMulticastTable.clear();
        switchTbl.insert(make_pair(switchKey, portMulticastTable));
    }

    switchIt = switchTbl.find(switchKey);
    PortMulticastTable &portTable = switchIt->second;

    PortKeyType portKey;
    portKey.nodeId = switchKey.nodeId;
    portKey.physicalId = switchKey.physicalId;
    portKey.portIndex = srcPortNum;

    PortMulticastTable::iterator portIt = portTable.find(portKey);

    if (portIt == portTable.end()) {
        PortMulticastInfo portMulticastInfo;
        memset(&portMulticastInfo, 0, sizeof(PortMulticastInfo));
        portMulticastInfo.portNum = srcPortNum;
        portTable.insert(make_pair(portKey, portMulticastInfo));
    }

    portIt = portTable.find(portKey);
    PortMulticastInfo &portInfo = portIt->second;

    std::vector<uint32_t> primaryReplicaDstPortList;
    std::vector<std::vector<uint32_t>> dstPortListGroups;
    primaryReplicaDstPortList.clear();
    dstPortListGroups.clear();

    portMaskTolist(primaryReplicaDstPorts, primaryReplicaDstPortList);
    uint32_t spraySize = primaryReplicaDstPortList.size();

    std::set<uint64_t>::iterator dstPortGroupsIt;
    for (dstPortGroupsIt = dstPortGroups.begin();
         dstPortGroupsIt != dstPortGroups.end();
         dstPortGroupsIt++) {

        uint64_t dstPortMask = *dstPortGroupsIt;
        std::vector<uint32_t> dstPortList;
        portMaskTolist(dstPortMask, dstPortList);

        FM_LOG_DEBUG("%s: mcId %d, source port %d/0x%x/%d, dstPortMask 0x%016lx, spraySize %d",
                    __FUNCTION__, groupInfo->mcId, switchKey.nodeId, switchKey.physicalId,
                    srcPortNum, dstPortMask, spraySize);

        if (dstPortList.size() > spraySize) {
            spraySize = dstPortList.size();
        }

        dstPortListGroups.push_back(dstPortList);
    }

    //  TODO add check to spraySize and mcSize

    PortMcIdTableEntry &mcIdEntry = portInfo.mcIdEntry;
    mcIdEntry.portNum = srcPortNum;
    mcIdEntry.extendedTable = false;
    mcIdEntry.index = groupInfo->mcId;
    mcIdEntry.extendedTablePtrValid = false;
    MulticastSprays &sprays = mcIdEntry.sprays;

    for (uint32_t i = 0; i < spraySize; i++) {
        std::ostringstream ss;
        ss << "    mcId " << groupInfo->mcId << " spray " << i << ":";

        // select a port from each portList
        MulticastPortList mcPortList;
        MulticastPort mcPort;
        mcPortList.replicaValid = false;
        mcPortList.replicaOffset = 0;

        // replica port goes as the first in the port list
        if ( primaryReplicaDstPortList.size() > 0 ) {
            mcPort.portNum = primaryReplicaDstPortList[i % primaryReplicaDstPortList.size()];
            mcPort.vcHop = 0;
            mcPortList.replicaValid = true;
            mcPortList.replicaOffset = 0;
            ss << " replica port: " << mcPort.portNum;
            mcPortList.ports.push_back(mcPort);
        }

        ss << " ports: ";
        for (uint32_t j = 0; j < dstPortListGroups.size(); j++) {
            std::vector<uint32_t> &dstPortList = dstPortListGroups[j];
            mcPort.portNum = dstPortList[i % dstPortList.size()];
            mcPort.vcHop = 0;
            mcPortList.ports.push_back(mcPort);
            ss << mcPort.portNum << " ";
        }

        ss << endl;
        FM_LOG_DEBUG("%s: %s, mcSize %d",
                    __FUNCTION__, ss.str().c_str(), (int)mcPortList.ports.size());

        // Add the spray
        sprays.push_back(mcPortList);
    }
}

// return true: if any GPU from this multicast group is reachable from this port
//       false: if no GPU from this multicast group is reachable from this port
bool
GlobalFmMulticastMgr::areGPUsReachableOnPort(MulticastGroupInfo *groupInfo, PortKeyType portKey)
{
    SwitchKeyType switchKey;
    switchKey.nodeId = portKey.nodeId;
    switchKey.physicalId = portKey.physicalId;

    GpusReachableFromPortMap::iterator it;
    it = mpGfm->mpParser->gpusReachableFromPortMap.find(switchKey);
    if ( it == mpGfm->mpParser->gpusReachableFromPortMap.end() ) {
        // No GPU reachable from this switch
        return false;
    }

    GpusReachableFromPort &gpusReachableFromPort = it->second;
    GpusReachableFromPort::iterator jit;
    jit = gpusReachableFromPort.find( portKey.portIndex );
    if ( jit == gpusReachableFromPort.end() ) {
        // No GPU is reachable from this port
        return false;
    }

    std::set<GpuKeyType>::iterator dstGpuKeyIt;

    for ( dstGpuKeyIt = groupInfo->gpuList.begin(); dstGpuKeyIt != groupInfo->gpuList.end(); dstGpuKeyIt++ ) {
        GpuKeyType gpuKey = *dstGpuKeyIt;
        uint32_t targetId;

        if ( !mpGfm->mpParser->getGpuTargetIdFromKey( gpuKey, targetId ) ) {
            FM_LOG_ERROR("Failed to find targetId for GPU " NODE_ID_LOG_STR " %d and physical id %d",
                         gpuKey.nodeId, gpuKey.physicalId);
            continue;
        }

        // GPUs that are reachable from this port
        GpuTargetIds &srcGpuSet = jit->second;

        if ( srcGpuSet.find(targetId) != srcGpuSet.end() ) {
            // at least one GPU is reachable
            return true;
        }
    }

    // no GPU in this multicast group is reachable from this port
    return false;
}

void
GlobalFmMulticastMgr::getDstPortMaskToGpu(PortKeyType srcPortKey, uint32_t dstTargetId, uint64_t &dstPortMask)
{
    dstPortMask = 0;

    std::map <RidTableKeyType, ridRouteEntry *>::iterator ridIt;
    RidTableKeyType ridKey;

    ridKey.nodeId = srcPortKey.nodeId;
    ridKey.physicalId = srcPortKey.physicalId;
    ridKey.portIndex = srcPortKey.portIndex;

    ridKey.index = dstTargetId;
    ridIt = mpGfm->mpParser->ridEntry.find(ridKey);

    if ( ridIt != mpGfm->mpParser->ridEntry.end() ) {
        //
        // a valid unicast rid entry is found to reach this GPU
        // return the dst port from this unicast rid entry
        //
        mpGfm->mpParser->ridEntryToDstPortMask( ridIt->second, dstPortMask );
        return;
    }

    //
    // no rid entry is found for this GPU, but the GPU,
    // but srcPortKey could be the access port that is directly connected to dstTargetId
    //
    accessPort *accessPortInfo = mpGfm->mpParser->getAccessPortInfo( srcPortKey.nodeId,
                                                                     srcPortKey.physicalId,
                                                                     srcPortKey.portIndex );
    if ( !accessPortInfo ) {
        // not an access port
        return;
    }

    GpuKeyType gpuKey;
    gpuKey.nodeId = accessPortInfo->has_farnodeid() ? accessPortInfo->farnodeid() : 0;
    gpuKey.physicalId = accessPortInfo->has_farpeerid() ? accessPortInfo->farpeerid() : 0;

    uint32_t targetId;
    if ( ( mpGfm->mpParser->getGpuTargetIdFromKey( gpuKey, targetId ) == false ) ||
         ( targetId != dstTargetId ) ) {
        return;
    }

    // dstTargetId is directly connected to access port srcPortKey, add source port to dstPortMask
    dstPortMask |= (1LL << srcPortKey.portIndex);
}

MulticastGroupTable::iterator
GlobalFmMulticastMgr::getMulticastGroupIterator(uint32_t partitionId, uint32_t groupId)
{
    MulticastGroupKeyType key;
    key.partitionId = partitionId;
    key.groupId = groupId;

    return mGroupTable.find(key);
}

MulticastGroupInfo *
GlobalFmMulticastMgr::getMulticastGroup(uint32_t partitionId, uint32_t groupId)
{
    MulticastGroupInfo *pGroupInfo = NULL;
    MulticastGroupTable::iterator it = getMulticastGroupIterator(partitionId, groupId);
    if ( it != mGroupTable.end() ) {
        pGroupInfo = it->second;
    }

    return pGroupInfo;
}

FMIntReturn_t
GlobalFmMulticastMgr::sendGroupCreateRspMsg(uint64_t mcHandle, uint32_t  createNodeId,
                                            lwswitch::configStatus rspCode)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupCreateResponse *pMsg = new lwswitch::multicastGroupCreateResponse();

    pFmMessage->set_type(lwswitch::FM_MULTICAST_GROUP_CREATE_RSP);
    pFmMessage->set_allocated_mcgroupcreatersp(pMsg);

    pMsg->set_mchandle(mcHandle);
    pMsg->set_createnodeid(createNodeId);
    pMsg->set_rspcode(rspCode);

    rc =  mpGfm->SendMessageToLfm(createNodeId, pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send multicast group create response to " NODE_ID_LOG_STR " %d for handle 0x%lu.",
                     createNodeId, mcHandle);
    }

    delete(pFmMessage);
    return rc;
}

FMIntReturn_t
GlobalFmMulticastMgr::sendGroupBindRspMsg(uint64_t mcHandle, uint32 createNodeId, uint32_t bindNodeId,
                                          lwswitch::configStatus rspCode)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupBindResponse *pMsg = new lwswitch::multicastGroupBindResponse();

    pFmMessage->set_type(lwswitch::FM_MULTICAST_GROUP_BIND_RSP);
    pFmMessage->set_allocated_mcgroupbindrsp(pMsg);

    pMsg->set_mchandle(mcHandle);
    pMsg->set_createnodeid(createNodeId);
    pMsg->set_rspcode(rspCode);

    rc =  mpGfm->SendMessageToLfm(bindNodeId, pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send multicast group bind response to " NODE_ID_LOG_STR " %d for handle 0x%lu.",
                     bindNodeId, mcHandle);
    }

    delete(pFmMessage);
    return rc;
}

FMIntReturn_t
GlobalFmMulticastMgr::sendGroupSetupCompleteReqMsg(uint64_t mcHandle, uint32_t createNodeId,
                                                   MulticastGroupInfo *groupInfo,
                                                   lwswitch::configStatus errCode)
{
    if (!groupInfo) {
        return FM_INT_ST_GENERIC_ERROR;
    }

    lwswitch::fmMessage *pFmMessage = NULL;
    lwswitch::multicastGroupSetupCompleteRequest *pMsg = NULL;

    uint64_t mappedAddr, mappedAddrRange, addrValid;
    uint64_t parentMappedAddr, parentMappedAddrRange, parentAddrValid;
    FMIntReturn_t rc;

    //
    // For the first allocation, parentMcId and mcId would be both valid and the same,
    // and allocated FLA and range would be the same for the parent and the sub group.
    //
    rc = getMulticastGroupBaseAddrAndRange(groupInfo->parentMcId, parentMappedAddr, parentMappedAddrRange);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to get parent group multicast address for group %d",
                     groupInfo->parentMcId);
        parentAddrValid = false;
    } else {
        parentAddrValid = true;
    }

    rc = getMulticastGroupBaseAddrAndRange(groupInfo->mcId, mappedAddr, mappedAddrRange);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to get multicast address for group %d",
                     groupInfo->mcId);
        addrValid = false;
    } else {
        addrValid = true;
    }

    // send the message to all nodes
    NodeMulticastTable::iterator it;
    for (it = groupInfo->multicastTable.begin(); it != groupInfo->multicastTable.end(); it++) {
        uint32_t nodeId = it->first;

        pFmMessage = new lwswitch::fmMessage();
        pMsg = new lwswitch::multicastGroupSetupCompleteRequest();

        pFmMessage->set_type(lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_REQ);
        pFmMessage->set_allocated_mcgroupsetupcompletereq(pMsg);

        pMsg->set_mchandle(mcHandle);
        pMsg->set_createnodeid(createNodeId);

        pMsg->set_parentmcflaaddrvalid(parentAddrValid);
        pMsg->set_mcflaaddrvalid(addrValid);

        if (!parentAddrValid || !addrValid) {
            errCode = lwswitch::FM_MC_GENERIC_ERROR;
        } else {
            pMsg->set_parentmcflaaddrbase(parentMappedAddr);
            pMsg->set_parentmcflaaddrrange(parentMappedAddrRange);

            pMsg->set_mcflaaddrbase(mappedAddr);
            pMsg->set_mcflaaddrrange(mappedAddrRange);
        }

        // include the GPUs on this node
        std::set<GpuKeyType>::iterator jit;
        for (jit = groupInfo->gpuList.begin(); jit != groupInfo->gpuList.end(); jit++) {
            const GpuKeyType &gpuKey = *jit;

            if (gpuKey.nodeId != nodeId) {
                // the gpu is not on this node, not need to include in this message
                continue;
            }

            char uuid[FM_UUID_BUFFER_SIZE];
            if (!mpGfm->getGpuUuid(gpuKey.nodeId, gpuKey.physicalId, uuid)) {
                FM_LOG_ERROR("failed to get GPU " NODE_ID_LOG_STR " %d physical id %d uuid",
                              gpuKey.nodeId, gpuKey.physicalId);
                errCode = lwswitch::FM_MC_GENERIC_ERROR;
                continue;
            }

            pMsg->add_uuid(uuid);
        }

        pMsg->set_errcode(errCode);
        rc =  mpGfm->SendMessageToLfm(nodeId, pFmMessage, false);
        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to send multicast group setup complete resquest to " NODE_ID_LOG_STR " %d for handle 0x%lu.",
                         nodeId, mcHandle);
        }

        delete(pFmMessage);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFmMulticastMgr::sendGroupReleaseRspMsg(uint64_t mcHandle, uint32 createNodeId, uint32_t releaseNodeId,
                                             lwswitch::configStatus rspCode)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupReleaseResponse *pMsg = new lwswitch::multicastGroupReleaseResponse();

    pFmMessage->set_type(lwswitch::FM_MULTICAST_GROUP_RELEASE_RSP);
    pFmMessage->set_allocated_mcgroupreleasersp(pMsg);

    pMsg->set_mchandle(mcHandle);
    pMsg->set_createnodeid(createNodeId);
    pMsg->set_rspcode(rspCode);

    rc =  mpGfm->SendMessageToLfm(releaseNodeId, pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send multicast group release response to " NODE_ID_LOG_STR " %d for handle 0x%lu.",
                     releaseNodeId, mcHandle);
    }

    delete(pFmMessage);
    return rc;
}

FMIntReturn_t
GlobalFmMulticastMgr::sendGroupReleaseCompleteReqMsg(uint64_t mcHandle, uint32_t createNodeId,
                                                     MulticastGroupInfo *groupInfo,
                                                     lwswitch::configStatus errCode)
{
    if (!groupInfo) {
        return FM_INT_ST_GENERIC_ERROR;
    }

    lwswitch::fmMessage *pFmMessage = NULL;
    lwswitch::multicastGroupReleaseCompleteRequest *pMsg = NULL;

    uint64_t mappedAddr, mappedAddrRange;
    FMIntReturn_t rc = getMulticastGroupBaseAddrAndRange(groupInfo->parentMcId, mappedAddr, mappedAddrRange);

    // send the message to all nodes
    NodeMulticastTable::iterator it;
    for (it = groupInfo->multicastTable.begin(); it != groupInfo->multicastTable.end(); it++) {
        uint32_t nodeId = it->first;

        pFmMessage = new lwswitch::fmMessage();
        pMsg = new lwswitch::multicastGroupReleaseCompleteRequest();

        pFmMessage->set_type(lwswitch::FM_MULTICAST_GROUP_RELEASE_COMPLETE_REQ);
        pFmMessage->set_allocated_mcgroupreleasecompletereq(pMsg);
        pMsg->set_mchandle(mcHandle);

        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to get parent group multicast address of group %d",
                         groupInfo->parentMcId);
            pMsg->set_parentmcflaaddrvalid(false);
            errCode = lwswitch::FM_MC_GENERIC_ERROR;
        } else {
            pMsg->set_parentmcflaaddrvalid(true);
            pMsg->set_parentmcflaaddrbase(mappedAddr);
            pMsg->set_parentmcflaaddrrange(mappedAddrRange);
        }

        rc = getMulticastGroupBaseAddrAndRange(groupInfo->mcId, mappedAddr, mappedAddrRange);
        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to get multicast address of group %d",
                         groupInfo->parentMcId);
            pMsg->set_mcflaaddrvalid(false);
            errCode = lwswitch::FM_MC_GENERIC_ERROR;
        } else {
            pMsg->set_mcflaaddrvalid(true);
            pMsg->set_mcflaaddrbase(mappedAddr);
            pMsg->set_mcflaaddrrange(mappedAddrRange);
        }

        // include the GPUs on this node
        std::set<GpuKeyType>::iterator jit;
        for (jit = groupInfo->gpuList.begin(); jit != groupInfo->gpuList.end(); jit++) {
            const GpuKeyType &gpuKey = *jit;

            if (gpuKey.nodeId != nodeId) {
                // the gpu is not on this node, not need to include in this message
                continue;
            }

            char uuid[FM_UUID_BUFFER_SIZE];
            if (!mpGfm->getGpuUuid(gpuKey.nodeId, gpuKey.physicalId, uuid)) {
                FM_LOG_ERROR("failed to get GPU " NODE_ID_LOG_STR " %d physical id %d uuid",
                              gpuKey.nodeId, gpuKey.physicalId);
                errCode = lwswitch::FM_MC_GENERIC_ERROR;
                continue;
            }

            pMsg->add_uuid(uuid);
        }

        pMsg->set_errcode(errCode);
        rc =  mpGfm->SendMessageToLfm(nodeId, pFmMessage, false);
        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to send multicast group release complete to " NODE_ID_LOG_STR " %d for handle 0x%lu.",
                         nodeId, mcHandle);
        }

        delete(pFmMessage);
    }

    return FM_INT_ST_OK;
}

void
GlobalFmMulticastMgr::handleGroupCreateReqMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO
    // allocate global handle, and send the handle back to lfm, which hands the handle to RM
}

void
GlobalFmMulticastMgr::handleGroupBindReqMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO
    // check and add the GPUs in the bind request to the group
    // when all GPUs have joined, allocate mcid and program the multicast routing tables
}

void
GlobalFmMulticastMgr::handleGroupSetupCompleteAckMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO
    // finalize the internal data structure for this group
}

void
GlobalFmMulticastMgr::handleGroupReleaseReqMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO
    // check and remove the GPUs in the release request from the group
    // when all GPUs have left, release the mcid and program the multicast routing tables
}

void
GlobalFmMulticastMgr::handleGroupReleaseCompleteAckMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO
    // clean up the internal data structure for this group
}
#endif
