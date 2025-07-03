/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <sstream>
#include <string.h>

#include "fm_log.h"
#include "GlobalFMLWLinkState.h"
#include "FMLWLinkDeviceRepo.h"
#include "GlobalFabricManager.h"
#include "GlobalFMLWLinkConnRepo.h"

FMLWLinkDetailedConnInfo::FMLWLinkDetailedConnInfo(FMLWLinkEndPointInfo masterEnd,
                                                   FMLWLinkEndPointInfo slaveEnd)
{
    mMasterEnd = masterEnd;
    mSlaveEnd = slaveEnd;
    memset( &mMasterLinkState, -1, sizeof(mMasterLinkState) );
    memset( &mSlaveLinkState, -1, sizeof(mSlaveLinkState) );
    memset( &mMasterLinkQualityInfo, 0, sizeof(mMasterLinkQualityInfo) );
    memset( &mSlaveLinkLinkQualityInfo, 0, sizeof(mSlaveLinkLinkQualityInfo) );
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    memset( &mMasterLinkFomValues, 0, sizeof(mMasterLinkFomValues));
    memset( &mSlaveLinkFomValues, 0, sizeof(mSlaveLinkFomValues));
    memset( &mMasterLinkGradingValues, 0, sizeof(mMasterLinkGradingValues));
    memset( &mSlaveLinkGradingValues, 0, sizeof(mSlaveLinkGradingValues));;
#endif
}

FMLWLinkDetailedConnInfo::~FMLWLinkDetailedConnInfo()
{
    // nothing specific
}

void
FMLWLinkDetailedConnInfo::setLinkStateInfo(FMLWLinkStateInfo masterInfo,
                                           FMLWLinkStateInfo slaveInfo)
{
    mMasterLinkState = masterInfo;
    mSlaveLinkState = slaveInfo;
}

void
FMLWLinkDetailedConnInfo::setLinkStateInfoMaster(FMLWLinkStateInfo masterInfo)
{
    mMasterLinkState = masterInfo;
}

void
FMLWLinkDetailedConnInfo::setLinkStateInfoSlave(FMLWLinkStateInfo slaveInfo)
{
    mSlaveLinkState = slaveInfo;
}

void
FMLWLinkDetailedConnInfo::setMasterLinkQualityInfo(FMLWLinkQualityInfo &linkQualityInfo)
{
    mMasterLinkQualityInfo = linkQualityInfo;
}
void
FMLWLinkDetailedConnInfo::setSlaveLinkQualityInfo(FMLWLinkQualityInfo &linkQualityInfo)
{
    mSlaveLinkLinkQualityInfo = linkQualityInfo;
}

bool
FMLWLinkDetailedConnInfo::isConnTrainedToActive(void)
{
    if ( linkStateActive() &&
         txSubLinkStateActive() &&
         rxSubLinkStateActive() ) {
        return true;
    }
    return false;
}

void
FMLWLinkDetailedConnInfo::dumpConnInfo(std::ostream *os, GlobalFabricManager *gfm,
                                       GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMLWLinkDevInfo masterDevInfo, slaveDevInfo;
    FMFabricParser *parser = gfm->mpParser;
    lwlink_pci_dev_info masterPciDevInfo, slavePciDevInfo;
    uint32 masterPhysicalId, slavePhysicalId;
    uint32_t masterOsfpPort = -1, slaveOsfpPort = -1;

    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    masterPciDevInfo = masterDevInfo.getDevicePCIInfo();
    slavePciDevInfo = slaveDevInfo.getDevicePCIInfo();

    if ((masterDevInfo.getDeviceType() == lwlink_device_type_lwswitch) &&
        (gfm->getLWSwitchPhysicalId(mMasterEnd.nodeId, masterPciDevInfo, masterPhysicalId)) &&
        (parser->isTrunkPort(mMasterEnd.nodeId, masterPhysicalId, mMasterEnd.linkIndex))) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        masterOsfpPort = parser->getOsfpPortNumForLinkIndex(mMasterEnd.linkIndex);
#endif
    }

    if ((slaveDevInfo.getDeviceType() == lwlink_device_type_lwswitch) &&
        (gfm->getLWSwitchPhysicalId(mSlaveEnd.nodeId, slavePciDevInfo, slavePhysicalId)) &&
        (parser->isTrunkPort(mSlaveEnd.nodeId, slavePhysicalId, mSlaveEnd.linkIndex))) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        slaveOsfpPort = parser->getOsfpPortNumForLinkIndex(mSlaveEnd.linkIndex);
#endif
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    *os << "NodeID:" << mMasterEnd.nodeId << " ";
#endif

    *os << " DeviceName:" << masterDevInfo.getDeviceName();
    if (masterDevInfo.getDeviceType() == lwlink_device_type_lwswitch)
        *os << " (PhysicalId:" << masterPhysicalId << ")";

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (masterOsfpPort != (uint32_t)-1)
        *os << " OsfpPort:" << masterOsfpPort;
#endif
    *os << " LWLinkIndex:" << mMasterEnd.linkIndex;

    *os << " <=======> ";

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    *os << "NodeID:" << mSlaveEnd.nodeId << " ";
#endif

    *os << " DeviceName:" << slaveDevInfo.getDeviceName();
    if (slaveDevInfo.getDeviceType() == lwlink_device_type_lwswitch)
        *os << " (PhysicalId:" << slavePhysicalId << ")";
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (slaveOsfpPort != (uint32_t)-1)
        *os << " OsfpPort:" << slaveOsfpPort;
#endif
    *os << " LWLinkIndex:" << mSlaveEnd.linkIndex << std::endl;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
FMLWLinkDetailedConnInfo::printFomAndGradingValues(GlobalFabricManager *gfm,
                                                   GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMLWLinkDevInfo masterDevInfo, slaveDevInfo;
    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    lwlink_pci_dev_info masterPciDevInfo = masterDevInfo.getDevicePCIInfo();
    lwlink_pci_dev_info slavePciDevInfo = slaveDevInfo.getDevicePCIInfo(); 

    uint32 masterPhysicalId, slavePhysicalId;
    gfm->getLWSwitchPhysicalId(mMasterEnd.nodeId, masterPciDevInfo, masterPhysicalId);
    gfm->getLWSwitchPhysicalId(mSlaveEnd.nodeId, slavePciDevInfo, slavePhysicalId);
 
    FMFabricParser *parser = gfm->mpParser;
    uint32_t masterOsfpPort = parser->getOsfpPortNumForLinkIndex(mMasterEnd.linkIndex);
    uint32_t slaveOsfpPort = parser->getOsfpPortNumForLinkIndex(mSlaveEnd.linkIndex);

    FM_LOG_INFO("Failed connection NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d LWLinkIndex:%d "
                "<=======> NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d LWLinkIndex:%d ", \
                mMasterEnd.nodeId, masterDevInfo.getDeviceName().c_str(), masterPhysicalId, masterOsfpPort, mMasterEnd.linkIndex,\
                mSlaveEnd.nodeId, slaveDevInfo.getDeviceName().c_str(), slavePhysicalId, slaveOsfpPort, mSlaveEnd.linkIndex);

    std::stringstream outStr;

    outStr << "\n";
    outStr << "FOM Values\n";
    outStr << "\tNumLanes:" << setw(2) << int(mMasterLinkFomValues.numLanes) << setw(9) << "Values:";
    for(int i =0; i < mMasterLinkFomValues.numLanes; i++)
        outStr << setw(3) << mMasterLinkFomValues.fomValues[i];
    outStr << "\t<======>\t";
    outStr << "NumLanes:" << setw(2) << int(mSlaveLinkFomValues.numLanes) << setw(9) << "Values:";
    for(int i =0; i < mSlaveLinkFomValues.numLanes; i++)
        outStr << setw(3) << mSlaveLinkFomValues.fomValues[i];
    outStr << "\n";

    outStr << "GradingValues\n";
    outStr << "\tLaneMask: 0x" << setw(2) << std::hex << int(mMasterLinkGradingValues.laneMask) << std::dec;
    outStr << "\t\t<======>\t";
    outStr << "LaneMask: 0x" << setw(2) << std::hex << int(mMasterLinkGradingValues.laneMask) << std::dec;
    outStr << "\n";

    outStr << "\tTxInit: ";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mMasterLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mMasterLinkGradingValues.txInit[i]);
    outStr << "\t<======>\t";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mSlaveLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mSlaveLinkGradingValues.txInit[i]);
    outStr << "\n";

    outStr << "\tRxInit: ";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mMasterLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mMasterLinkGradingValues.rxInit[i]);
    outStr << "\t<======>\t";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mSlaveLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mSlaveLinkGradingValues.rxInit[i]);
    outStr << "\n";

    outStr << "\tTxMaint:";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mMasterLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mMasterLinkGradingValues.txMaint[i]);
    outStr << "\t<======>\t";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mSlaveLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mSlaveLinkGradingValues.txMaint[i]);
    outStr << "\n";

    outStr << "\tRxMaint:";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mMasterLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mMasterLinkGradingValues.rxMaint[i]);
    outStr << "\t<======>\t";
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++)
        if ((mSlaveLinkGradingValues.laneMask >> i) & 0x1)
            outStr << setw(3) << int(mSlaveLinkGradingValues.rxMaint[i]);
    outStr << "\n";
    FM_LOG_INFO("%s", outStr.str().c_str());
}
#endif

void
FMLWLinkDetailedConnInfo::dumpConnAndStateInfo(std::ostream *os, GlobalFabricManager *gfm,
                                               GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMLWLinkDevInfo masterDevInfo, slaveDevInfo;
    FMFabricParser *parser = gfm->mpParser;
    lwlink_pci_dev_info masterPciDevInfo, slavePciDevInfo;
    uint32 masterPhysicalId, slavePhysicalId;
    uint32_t masterOsfpPort = -1, slaveOsfpPort = -1;

    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    masterPciDevInfo = masterDevInfo.getDevicePCIInfo();
    slavePciDevInfo = slaveDevInfo.getDevicePCIInfo();

    if ((masterDevInfo.getDeviceType() == lwlink_device_type_lwswitch) &&
        (gfm->getLWSwitchPhysicalId(mMasterEnd.nodeId, masterPciDevInfo, masterPhysicalId)) &&
        (parser->isTrunkPort(mMasterEnd.nodeId, masterPhysicalId, mMasterEnd.linkIndex))) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        masterOsfpPort = parser->getOsfpPortNumForLinkIndex(mMasterEnd.linkIndex);
#endif
    }

    if ((slaveDevInfo.getDeviceType() == lwlink_device_type_lwswitch) &&
        (gfm->getLWSwitchPhysicalId(mSlaveEnd.nodeId, slavePciDevInfo, slavePhysicalId)) &&
        (parser->isTrunkPort(mSlaveEnd.nodeId, slavePhysicalId, mSlaveEnd.linkIndex))) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        slaveOsfpPort = parser->getOsfpPortNumForLinkIndex(mSlaveEnd.linkIndex);
#endif
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    *os << "NodeID:" << mMasterEnd.nodeId << " ";
#endif

    *os << " DeviceName:" << masterDevInfo.getDeviceName();
    if (masterDevInfo.getDeviceType() == lwlink_device_type_lwswitch)
        *os << " (PhysicalId:" << masterPhysicalId << ")";

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (masterOsfpPort != (uint32_t)-1)
        *os << " OsfpPort:" << masterOsfpPort;
#endif
    *os << " LWLinkIndex:" << mMasterEnd.linkIndex;

    *os << " <=======> ";

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    *os << "NodeID:" << mSlaveEnd.nodeId << " ";
#endif

    *os << " DeviceName:" << slaveDevInfo.getDeviceName();
    if (slaveDevInfo.getDeviceType() == lwlink_device_type_lwswitch)
        *os << " (PhysicalId:" << slavePhysicalId << ")";
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (slaveOsfpPort != (uint32_t)-1)
        *os << " OsfpPort:" << slaveOsfpPort;
#endif
    *os << " LWLinkIndex:" << mSlaveEnd.linkIndex << std::endl;

    // dump link state
    *os << "LWLink_State:" << GlobalFMLWLinkState::getMainLinkState( mMasterLinkState.linkMode );
    *os << " Tx_State:" << GlobalFMLWLinkState::getTxSubLinkState( mMasterLinkState.txSubLinkMode );
    *os << " Rx_State:" << GlobalFMLWLinkState::getRxSubLinkState( mMasterLinkState.rxSubLinkMode );
    *os << " <=======> ";
    *os << "LWLink_State:" << GlobalFMLWLinkState::getMainLinkState( mSlaveLinkState.linkMode );
    *os << " Tx_State:" << GlobalFMLWLinkState::getTxSubLinkState( mSlaveLinkState.txSubLinkMode );
    *os << " Rx_State:" << GlobalFMLWLinkState::getRxSubLinkState( mSlaveLinkState.rxSubLinkMode ) << std::endl;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
bool
FMLWLinkDetailedConnInfo::connectionPortInfoExists(ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                                   std::vector<int> portInfo, std::vector<int> farPortInfo)
{
    bool found = false;

    if (discoveredConnPortInfoMap.find(portInfo) != discoveredConnPortInfoMap.end() &&
        discoveredConnPortInfoMap[portInfo] == farPortInfo) {
        found = true;
    }

    if (discoveredConnPortInfoMap.find(farPortInfo) != discoveredConnPortInfoMap.end() &&
        discoveredConnPortInfoMap[farPortInfo] == portInfo) {
        found = true;
    }

    return found;
}

bool
FMLWLinkDetailedConnInfo::checkConnectionInconsistency(ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                                       std::vector<int> portInfo, std::vector<int> farPortInfo)
{
    bool found = false;

    if (discoveredConnPortInfoMap.find(portInfo) != discoveredConnPortInfoMap.end() &&
        discoveredConnPortInfoMap[portInfo] != farPortInfo) {
        FM_LOG_ERROR("incosistency detected as " NODE_ID_LOG_STR " %d, port number %d has multiple connections",
                     portInfo[NODEID], portInfo[PORTNUM]);
        found = true;
    }

    if (discoveredConnPortInfoMap.find(farPortInfo) != discoveredConnPortInfoMap.end() &&
        discoveredConnPortInfoMap[farPortInfo] != portInfo) {
        FM_LOG_ERROR("incosistency detected as " NODE_ID_LOG_STR " %d, port number %d has multiple connections",
                     farPortInfo[NODEID], farPortInfo[PORTNUM]);
        found = true;
    }

    return found;
}

void
FMLWLinkDetailedConnInfo::dumpDiscoveredConnectionPortInfo(GlobalFabricManager *gfm,
                                                           ConnectionPortInfoMap &discoveredConnPortInfoMap,
                                                           GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMLWLinkDevInfo masterDevInfo;
    FMLWLinkDevInfo slaveDevInfo;
    lwlink_pci_dev_info masterPciDevInfo;
    lwlink_pci_dev_info slavePciDevInfo;
    uint32 masterSwitchPhysicalId;
    uint32 slaveSwitchPhysicalId;
    uint32_t masterSlotId;
    uint32_t slaveSlotId;

    linkDevRepo.getDeviceInfo( mMasterEnd.nodeId, mMasterEnd.gpuOrSwitchId, masterDevInfo );
    linkDevRepo.getDeviceInfo( mSlaveEnd.nodeId, mSlaveEnd.gpuOrSwitchId, slaveDevInfo );

    masterPciDevInfo = masterDevInfo.getDevicePCIInfo();
    slavePciDevInfo = slaveDevInfo.getDevicePCIInfo();

    if (!gfm->getLWSwitchPhysicalId(mMasterEnd.nodeId, masterPciDevInfo, masterSwitchPhysicalId)) {
        // log error
        FM_LOG_ERROR("unable to find physical id for LWSwitch with " NODE_ID_LOG_STR " %d, pciBusId %s", 
                     mMasterEnd.nodeId, masterDevInfo.getDevicePciBusId());
        return;
    }

    if (!gfm->getLWSwitchPhysicalId(mSlaveEnd.nodeId, slavePciDevInfo, slaveSwitchPhysicalId)) {
        //log error
        FM_LOG_ERROR("unable to find physical id for LWSwitch with " NODE_ID_LOG_STR " %d, pciBusId %s", 
                     mSlaveEnd.nodeId, slaveDevInfo.getDevicePciBusId());
        return;
    }

    if (!gfm->mpParser->getSlotId(masterSwitchPhysicalId, masterSlotId)) {
        FM_LOG_ERROR("unable to find slot id for LWSwitch with node id %d, physicalId %d", 
                     mMasterEnd.nodeId, masterSwitchPhysicalId);
        return;
    }

    if (!gfm->mpParser->getSlotId(slaveSwitchPhysicalId, slaveSlotId)) {
        FM_LOG_ERROR("unable to find slot id for LWSwitch with node id %d, physicalId %d", 
                     mMasterEnd.nodeId, slaveSwitchPhysicalId);
        return;
    }

    // fill discoveredConnPortInfoMap
    std::vector<int> portInfo;
    std::vector<int> farPortInfo;
    portInfo.push_back(mMasterEnd.nodeId);
    // the corresponding min
    portInfo.push_back(masterSlotId);
    portInfo.push_back(gfm->mpParser->getOsfpPortNumForLinkIndex(mMasterEnd.linkIndex));
    farPortInfo.push_back(mSlaveEnd.nodeId);
    farPortInfo.push_back(slaveSlotId);
    farPortInfo.push_back(gfm->mpParser->getOsfpPortNumForLinkIndex(mSlaveEnd.linkIndex));

    if (checkConnectionInconsistency(discoveredConnPortInfoMap, portInfo, farPortInfo)) {
        return;
    }

    if (connectionPortInfoExists(discoveredConnPortInfoMap, portInfo, farPortInfo)) {
        return;
    }
    
    discoveredConnPortInfoMap.insert(make_pair(portInfo, farPortInfo));
}
#endif

bool
FMLWLinkDetailedConnInfo::linkStateActive(void)
{
    if ( (mMasterLinkState.linkMode == lwlink_link_mode_active) &&
         (mSlaveLinkState.linkMode == lwlink_link_mode_active) ) {
        return true;
    }

    return false;
}

bool
FMLWLinkDetailedConnInfo::isConnInContainState(void) {
    if (mMasterLinkState.linkMode == lwlink_link_mode_contain || 
        mSlaveLinkState.linkMode == lwlink_link_mode_contain) {
        return true;
    }
    return false;
}

bool
FMLWLinkDetailedConnInfo::txSubLinkStateActive(void)
{
    if ( ((mMasterLinkState.txSubLinkMode == lwlink_tx_sublink_mode_hs) ||
          (mMasterLinkState.txSubLinkMode == lwlink_tx_sublink_mode_single_lane)) &&
         ((mSlaveLinkState.txSubLinkMode == lwlink_tx_sublink_mode_hs) ||
          (mSlaveLinkState.txSubLinkMode == lwlink_tx_sublink_mode_single_lane)) ) {
        return true;
    }

    return false;
}

bool
FMLWLinkDetailedConnInfo::rxSubLinkStateActive(void)
{
    if ( ((mMasterLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_hs) ||
          (mMasterLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_single_lane)) &&
         ((mSlaveLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_hs) ||
          (mSlaveLinkState.rxSubLinkMode == lwlink_rx_sublink_mode_single_lane)) ) {
        return true;
    }

    return false;
}

GlobalFMLWLinkConnRepo::GlobalFMLWLinkConnRepo()
{
    mIntraConnMap.clear();
    mInterConnMap.clear();
}

GlobalFMLWLinkConnRepo::~GlobalFMLWLinkConnRepo()
{
    clearConns();
}

void
GlobalFMLWLinkConnRepo::clearConns()
{
    // delete all the intra-node connections
    LWLinkIntraConnMap::iterator it = mIntraConnMap.begin();
    while ( it != mIntraConnMap.end() ) {
        FMLWLinkDetailedConnInfoList &connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit = connList.begin();
        while ( jit != connList.end() ) {
            FMLWLinkDetailedConnInfo *conn = (*jit);
            connList.erase( jit++ );
            delete conn;
        }
        mIntraConnMap.erase( it++ );
    }

    // delete all the inter-node connections
    LWLinkInterNodeConns::iterator jit =  mInterConnMap.begin();
    while ( jit != mInterConnMap.end() ) {
        FMLWLinkDetailedConnInfo *conn = (*jit);
        mInterConnMap.erase( jit++ );
        delete conn;
    }
}

bool
GlobalFMLWLinkConnRepo::addIntraConnections(uint32 nodeId, FMLWLinkConnList &connList)
{
    FMLWLinkDetailedConnInfoList detailedConnList;
    FMLWLinkDetailedConnInfo* detailedConnInfo;

    // bail if we already have connection information for the specified node
    LWLinkIntraConnMap::iterator it = mIntraConnMap.find( nodeId );
    if ( it != mIntraConnMap.end() ) {
        return false;
    }

    FMLWLinkConnList::iterator jit;
    for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
        FMLWLinkConnInfo connInfo = (*jit);
        // create our fm detailed connection information
        detailedConnInfo = new FMLWLinkDetailedConnInfo( connInfo.masterEnd, connInfo.slaveEnd );
        detailedConnList.push_back( detailedConnInfo );
    }

    // add node's device information locally
    mIntraConnMap.insert( std::make_pair(nodeId, detailedConnList) );
    return true;
}

bool
GlobalFMLWLinkConnRepo::addInterConnections(FMLWLinkConnInfo connInfo)
{
    FMLWLinkDetailedConnInfo *detailedConnInfo;
    detailedConnInfo = new FMLWLinkDetailedConnInfo( connInfo.masterEnd, connInfo.slaveEnd );

    // before inserting, check for existing connection
    LWLinkInterNodeConns::iterator it;
    for ( it = mInterConnMap.begin(); it != mInterConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfo *tempConnInfo = (*it);
        if ( tempConnInfo == detailedConnInfo ) {
            // found the connection, with same endPoint info
            delete detailedConnInfo;
            return false;
        }
    }

    // no such connection exists for us, add it
    mInterConnMap.push_back( detailedConnInfo );
    return true;
}

bool
GlobalFMLWLinkConnRepo::mergeIntraConnections(uint32 nodeId, FMLWLinkDetailedConnInfoList newConnList)
{
    LWLinkIntraConnMap::iterator it = mIntraConnMap.find( nodeId );
    if ( it == mIntraConnMap.end() ) {
        // connection map for the node should already exist before merging
        FM_LOG_ERROR("LWLink connection map for " NODE_ID_LOG_STR " %d not found while trying to merge connections", nodeId);
        return false;
    } else {

        // add connections if it not already present and if presents,
        // update its connection state with new information.
        FMLWLinkDetailedConnInfoList &existingConnList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;

        for ( jit = newConnList.begin(); jit != newConnList.end(); jit++ ) {
            FMLWLinkDetailedConnInfo *newConnInfo = *jit;
            // create detailed connection information
            FMLWLinkDetailedConnInfo* newDetailedConnInfo;
            newDetailedConnInfo= new FMLWLinkDetailedConnInfo(newConnInfo->getMasterEndPointInfo(),
                                                              newConnInfo->getSlaveEndPointInfo());

            FMLWLinkDetailedConnInfoList::iterator eit;
            for (eit = existingConnList.begin(); eit != existingConnList.end(); eit++ ) {
                FMLWLinkDetailedConnInfo *existingDetailConnInfo = *eit;
                if (*existingDetailConnInfo == *newDetailedConnInfo) {
                    // the conn already exist, only need to update Link state
                    delete newDetailedConnInfo;
                    newDetailedConnInfo = NULL;

                    existingDetailConnInfo->setLinkStateInfo(newConnInfo->getMasterLinkStateInfo(),
                                                             newConnInfo->getSlaveLinkStateInfo());
                    break;
                }
            }

            if (newDetailedConnInfo != NULL) {
                // add the new conn
                newDetailedConnInfo->setLinkStateInfo(newConnInfo->getMasterLinkStateInfo(),
                                                      newConnInfo->getSlaveLinkStateInfo());
                existingConnList.push_back( newDetailedConnInfo );
            }
        }
    }

    return true;
}

FMLWLinkDetailedConnInfo*
GlobalFMLWLinkConnRepo::getConnectionInfo(FMLWLinkEndPointInfo &endInfo)
{
    // first check in intra connection list,
    LWLinkIntraConnMap::iterator it = mIntraConnMap.find( endInfo.nodeId );
    if ( it != mIntraConnMap.end() ) {
        FMLWLinkDetailedConnInfoList::iterator jit;
        FMLWLinkDetailedConnInfoList connList = it->second;
        for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
            FMLWLinkDetailedConnInfo *tempConnInfo = (*jit);
            // check whether source or dest endPoint of the connection match to
            // the given endPoint for look-up
            if  ( (tempConnInfo->getMasterEndPointInfo() == endInfo) ||
                  (tempConnInfo->getSlaveEndPointInfo() == endInfo) ) {
                // found the connection, return it
                return tempConnInfo;
            }
        }
    }

    // check in inter connection list as well.
    LWLinkInterNodeConns::iterator jit;
    for ( jit = mInterConnMap.begin(); jit != mInterConnMap.end(); jit++ ) {
        FMLWLinkDetailedConnInfo *tempConnInfo = (*jit);
        // check whether source or dest endPoint of the connection match to
        // the given endPoint for look-up
        if  ( (tempConnInfo->getMasterEndPointInfo() == endInfo) ||
              (tempConnInfo->getSlaveEndPointInfo() == endInfo) ) {
            // found the connection, return it
            return tempConnInfo;
        }
    }

    // not found in both intra and inter connection list
    return NULL;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
GlobalFMLWLinkConnRepo::dumpAllDiscoveredConnectionPortInfo(GlobalFabricManager *gfm,
                                                            GlobalFMLWLinkDevRepo &linkDevRepo)
{
    LWLinkInterNodeConns::iterator jit;
    std::ofstream discoveredConnFile;
    std::map<uint32_t, uint32_t> trunkLwlinkToOsfpPortMap;
    discoveredConnFile.open(MULTI_NODE_LWLINK_CONN_DUMP_FILE, std::ofstream::trunc | std::ofstream::out);
    //fill the header part
    std::stringstream discoveredConnFileHdr;
    discoveredConnFileHdr << "nodeId,slotId,portNumber," 
                          << "nodeIdFar,slotIdFar,portNumberFar" << std::endl;
    discoveredConnFile << discoveredConnFileHdr.str();

    ConnectionPortInfoMap discoveredConnPortInfoMap;

    for (jit = mInterConnMap.begin(); jit != mInterConnMap.end(); jit++) {
        FMLWLinkDetailedConnInfo *connInfo = (*jit);
        connInfo->dumpDiscoveredConnectionPortInfo(gfm, discoveredConnPortInfoMap, linkDevRepo);
    }

    // now the discoveredConnPortInfoMap has a sorted version of the trunk link connections
    // which is expected.
    ConnectionPortInfoMap::iterator it;
    for (it = discoveredConnPortInfoMap.begin(); it != discoveredConnPortInfoMap.end(); it++) {
        std::stringstream outStr;
        vector<int> nearPortInfo = it->first;
        vector<int> farPortInfo = it->second;
        outStr << std::setfill('0') << std::setw(3) << nearPortInfo[0] << ",";
        outStr << std::setfill('0') << std::setw(2) << nearPortInfo[1] << ",";
        outStr << std::setfill('0') << std::setw(3) << nearPortInfo[2] << ",";
        outStr << std::setfill('0') << std::setw(3) << farPortInfo[0] << ",";
        outStr << std::setfill('0') << std::setw(2) << farPortInfo[1] << ",";
        outStr << std::setfill('0') << std::setw(3) << farPortInfo[2] << std::endl;
        discoveredConnFile << outStr.str();
    }

    discoveredConnFile.close();
}
#endif

void
GlobalFMLWLinkConnRepo::dumpAllConnAndStateInfo(GlobalFabricManager *gfm,
                                                GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FM_LOG_INFO( "Dumping all the detected LWLink connections" );

    LWLinkIntraConnMap::iterator it;
    for ( it = mIntraConnMap.begin(); it != mIntraConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfoList connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;
        FM_LOG_INFO( "Total number of LWLink connections:%ld", connList.size() );
        // dump each connection information
        for (jit = connList.begin(); jit != connList.end(); jit++ ) {
            std::stringstream outStr;
            FMLWLinkDetailedConnInfo *connInfo = (*jit);
            connInfo->dumpConnAndStateInfo( &outStr, gfm, linkDevRepo);
            FM_LOG_INFO( "\n%s", outStr.str().c_str() );
        }
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FM_LOG_INFO( "Dumping all the detected LWLink internode connections" );
    FM_LOG_INFO( "Total number of LWLink internode connections:%ld", mInterConnMap.size() );
    LWLinkInterNodeConns::iterator jit;
    for (jit = mInterConnMap.begin(); jit != mInterConnMap.end(); jit++ ) {
        std::stringstream outStr;
        FMLWLinkDetailedConnInfo *connInfo = (*jit);
        connInfo->dumpConnAndStateInfo( &outStr, gfm, linkDevRepo);
        FM_LOG_INFO( "\n%s", outStr.str().c_str() );
    }
#endif    
}
