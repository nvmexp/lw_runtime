/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <string.h>

#include "fm_log.h"
#include "FMErrorCodesInternal.h"
#include "lwlink_errors.h"
#include "FMLWLinkDeviceRepo.h"
#include "GlobalFabricManager.h"
#include "FMTopologyValidator.h"

FMLWLinkDevInfo::FMLWLinkDevInfo()
{
    mDeviceId = 0;
    memset(mDevUuid, 0, LWLINK_UUID_LEN);
    mNumLinks = 0;
    mDevType = 0;
    mEnabledLinkMask = 0;
    mNodeId = 0;
    mPciInfo = {0};

    mLinkInitStatus.clear();
    mLinkStateInfo.clear();
}

FMLWLinkDevInfo::FMLWLinkDevInfo(uint32 nodeId, lwlink_detailed_dev_info devInfo)
{
    // all the device information is received from LWLink driver in LFM context.
    // create device object based on driver returned format.

    mPciInfo = devInfo.pciInfo;

    memcpy( mDevUuid, devInfo.devUuid, sizeof(mDevUuid) );
    mDeviceName = std::string (devInfo.deviceName );
    mNumLinks = devInfo.numLinks;
    mDevType = devInfo.devType;
    mEnabledLinkMask = devInfo.enabledLinkMask;

    // create a unique id for this device. this id is per connection scope, i.e.
    // every time when local FM is connected to global GM, the device information
    // is exchanged along with this id and from then onwards global FM
    // can use this id to address each device in this node.

    // TODO - Fix this or tune this.
    //16bit domain, 8 bit bus, 8 bit device, 8 bit function
    uint64 deviceId = 0;
    deviceId = (uint64)(mPciInfo.domain) << 48;
    deviceId = deviceId | (uint64)mPciInfo.bus << 40;
    deviceId = deviceId | (uint64)mPciInfo.device << 32;
    deviceId = deviceId | (uint64)(mPciInfo.function) << 24;

    mDeviceId = deviceId;
    mNodeId = nodeId;
    mLinkInitStatus.clear();
    mLinkStateInfo.clear();

    populatePciBusIdInformation();
}

FMLWLinkDevInfo::FMLWLinkDevInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoMsg &devInfoMsg)
{
    // all the LWLink device information is received from peer LFM
    // create the object based on the message
    const lwswitch::devicePciInfo& pciInfoMsg = devInfoMsg.pciinfo();
    mPciInfo.domain = pciInfoMsg.domain();
    mPciInfo.bus = pciInfoMsg.bus();
    mPciInfo.device = pciInfoMsg.device();
    mPciInfo.function = pciInfoMsg.function();
    if ( devInfoMsg.has_uuid() ) {
        memcpy( mDevUuid, devInfoMsg.uuid().c_str(), sizeof(mDevUuid) );
    }

    mNodeId = nodeId;
    mDeviceId = devInfoMsg.deviceid();
    mDeviceName = devInfoMsg.devicename();
    mNumLinks = devInfoMsg.numlinks();
    mDevType = devInfoMsg.devtype();
    mEnabledLinkMask = devInfoMsg.enabledlinkmask();
    mLinkInitStatus.clear();
    mLinkStateInfo.clear();

    populatePciBusIdInformation();
}


void
FMLWLinkDevInfo::setLinkInitStatus(FMLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN])
{
    // first clear any existing state
    mLinkInitStatus.clear();
    for( uint32 idx = 0; idx < LWLINK_MAX_DEVICE_CONN; idx++ ) {
        FMLinkInitInfo tempStatus;
        tempStatus.initStatus = initStatus[idx].initStatus;
        tempStatus.linkIndex = initStatus[idx].linkIndex;
        mLinkInitStatus.push_back( tempStatus );
    }
}

bool
FMLWLinkDevInfo::getLinkState(uint32 linkIndex, FMLWLinkStateInfo &linkState)
{
    if (linkIndex >= LWLINK_MAX_DEVICE_CONN) {
        return false;
    }

    if (mLinkStateInfo.find(linkIndex) == mLinkStateInfo.end()) {
        // the link state info not found
        return false;
    }

    linkState = mLinkStateInfo[linkIndex];
    return true;
}

bool
FMLWLinkDevInfo::setLinkState(uint32 linkIndex, FMLWLinkStateInfo linkState)
{
    if (linkIndex >= LWLINK_MAX_DEVICE_CONN) {
        return false;
    }
    //
    // if we already have some state, remove it to get updated one
    // linkIndex is the map key. just make a blind attempt to erase it.
    //
    mLinkStateInfo.erase(linkIndex);
    mLinkStateInfo.insert( std::make_pair(linkIndex, linkState) );
    return true;
}

void
FMLWLinkDevInfo::getInitFailedLinksIndex(std::list<uint32> &initFailedLinks)
{
    // Link initialization status from LWLinkCoreLib driver is conselwtive. So
    // we need a contiguous array index while referencing it as some links (linkIdx)
    // may not be enabled.
    uint32 initIdx = 0;

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) )
            continue;

        FMLinkInitInfo tempStatus = mLinkInitStatus[initIdx];
        if ( tempStatus.initStatus == false ) {
            initFailedLinks.push_back(tempStatus.linkIndex);
        }
        initIdx++;
    }
}

void
FMLWLinkDevInfo::getNonActiveLinksIndex(std::list<uint32> &nonActiveLinks, std::list<uint32> &missingConnLinkIndex)
{

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) ) {
            continue;
        }

        if (mLinkStateInfo.find(linkIdx) == mLinkStateInfo.end()) {
            //link state info not found, connection missing
            // NOTE: this will include all trunk links that are never trained for LWSwitch
            nonActiveLinks.push_back(linkIdx);
            missingConnLinkIndex.push_back(linkIdx);
            continue;
        }

        FMLWLinkStateInfo tempStateInfo = mLinkStateInfo[linkIdx];
        if (tempStateInfo.linkMode != lwlink_link_mode_active)
        {
            nonActiveLinks.push_back(linkIdx);
        }
    }
}

//
// this function gives a list of links connected to the GPU that have actually failed
// ie, does not give the list of links which are in contain due to neighboring links failing.
// Actual failed links could be links that are not active, or links that are in contain mode
// and their sublink states are OFF.
//
void
FMLWLinkDevInfo::getFailedNonContainLinksIndex(std::list<uint32> &failedLinks, std::list<uint32> &missingConnLinkIndex)
{
    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) ) {
            continue;
        }

        if (mLinkStateInfo.find(linkIdx) == mLinkStateInfo.end()) {
            //link state info not found, connection missing
            // NOTE: this will include all trunk links that are never trained for LWSwitch
            failedLinks.push_back(linkIdx);
            missingConnLinkIndex.push_back(linkIdx);
            continue;
        }

        // links that go into contain mode due to errors from neighboring links
        // need not be added to list of failed links
        // the reason we check for the sublink modes as well is that it was observed
        // that for the actual failed link, the sublink states were in off, while for the
        // links that went into contain because of the actual failed link, the sublink
        // states were high speed/single lane
        FMLWLinkStateInfo tempStateInfo = mLinkStateInfo[linkIdx];
        if (tempStateInfo.linkMode != lwlink_link_mode_active &&
           !(tempStateInfo.linkMode == lwlink_link_mode_contain &&
            tempStateInfo.txSubLinkMode != lwlink_tx_sublink_mode_off &&
            tempStateInfo.rxSubLinkMode != lwlink_rx_sublink_mode_off))
        {
            failedLinks.push_back(linkIdx);
        }
    }
}

bool
FMLWLinkDevInfo::isAllLinksInContain()
{
    //MIG case should return false
    if (mEnabledLinkMask == 0) {
        return false;
    }

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) ) {
            continue;
        }

        FMLWLinkStateInfo tempStateInfo = mLinkStateInfo[linkIdx];
        if (tempStateInfo.linkMode != lwlink_link_mode_contain) {
            return false;
        }
    }

    return true;
}

bool
FMLWLinkDevInfo::isAllLinksInitialized(void)
{
    // Link initialization status from LWLinkCoreLib driver is conselwtive. So
    // we need a contiguous array index while referencing it as some links (linkIdx)
    // may not be enabled.
    uint32 initIdx = 0;

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) )
            continue;

        FMLinkInitInfo tempStatus = mLinkInitStatus[initIdx];
        if ( tempStatus.initStatus == false ) {
            // not all the links are initialized
            return false;
        }
        initIdx++;
    }
    return true;
}

bool
FMLWLinkDevInfo::isLinkActive(uint32_t linkIndex)
{
    // the link is not enabled
    if ( !(mEnabledLinkMask & ((uint64)1 << linkIndex)) ) {
        return false;
    }

    if (mLinkStateInfo.find(linkIndex) == mLinkStateInfo.end()) {
        // the link state info not found
        return false;
    }

    FMLWLinkStateInfo tempStateInfo = mLinkStateInfo[linkIndex];
    if (tempStateInfo.linkMode != lwlink_link_mode_active) {
        // the link state is not active
        return false;
    }

    // the link is active
    return true;
}

bool
FMLWLinkDevInfo::getNumActiveLinks(uint32_t &numActiveLinks)
{
    numActiveLinks = 0;

    for ( uint32 idx = 0; idx< LWLINK_MAX_DEVICE_CONN; idx++ ) {
        // all the enabled GPU links are supposed to be connected for bare metal
        // and VM configuration. So skipping cross checking with topology information.
        if ( !(mEnabledLinkMask & ((uint64)1 << idx)) ) {
            // this link is disabled in RM.
            continue;
        }
        // enabled link, check its state
        FmPerLinkStateInfo::iterator it = mLinkStateInfo.find( idx );
        if ( it == mLinkStateInfo.end() ) {
            FM_LOG_INFO("GPU link state information is missing for " NODE_ID_LOG_STR " %d GPU pci bus id %s link index %d",
                        mNodeId, mPciBusId, idx);
            // skip this link as well
            continue;
        }

        FMLWLinkStateInfo linkState = it->second;
        if ( linkState.linkMode == lwlink_link_mode_active ) {
            // found an active link, update our count
            numActiveLinks++;
        }
    }

    return true;
}

void
FMLWLinkDevInfo::populatePciBusIdInformation()
{
    FMPciInfo_t pciInfo;

    pciInfo.domain = mPciInfo.domain;
    pciInfo.bus = mPciInfo.bus;
    pciInfo.device = mPciInfo.device;
    pciInfo.function = mPciInfo.function; //RM don't provide this information.
    snprintf(mPciBusId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
             FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&pciInfo));
}

void
FMLWLinkDevInfo::dumpInfo(std::ostream *os)
{
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    *os << "\t Node ID:" << std::dec << mNodeId << std::endl;
#endif
    *os << "\t Device Name:" << mDeviceName << std::endl;
    *os << "\t PCI Bus ID:" << mPciBusId << std::endl;
    *os << "\t Device Type:" << std::dec << mDevType << std::endl;
    *os << "\t Number of Links:" << std::dec <<  mNumLinks << std::endl;
    *os << "\t Link Initialization Status" << std::endl;
    for ( uint32 idx = 0; idx < mNumLinks; idx++) {
        FMLinkInitInfo tempStatus = mLinkInitStatus[idx];
        if ( tempStatus.initStatus ) {
            *os << "\t\t\t Link Index " << std::dec << tempStatus.linkIndex << ":Success" << std::endl;
        } else {
            *os << "\t\t\t Link Index " << std::dec << tempStatus.linkIndex << ":Failed" << std::endl;
        }
    }
}

FMLWLinkDevInfo::~FMLWLinkDevInfo()
{
    // free the memory in the list and the map
    mLinkInitStatus.clear();
    mLinkStateInfo.clear();
}

LocalFMLWLinkDevRepo::LocalFMLWLinkDevRepo(LocalFMLWLinkDrvIntf *linkDrvIntf)
{
    // query the lwlink driver and populate all the registered device information
    mLWLinkDevList.clear();
    mLWLinkDrvIntf = linkDrvIntf;
    mLocalNodeId = 0;
}

LocalFMLWLinkDevRepo::~LocalFMLWLinkDevRepo()
{
    // update our local device information
    mLWLinkDevList.clear();
}

lwlink_pci_dev_info
LocalFMLWLinkDevRepo::getDevicePCIInfo(uint64 deviceId)
{
    // return empty values if not found
    lwlink_pci_dev_info pciInfo = {0};
    FMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();

    while ( it != mLWLinkDevList.end() ) {
        FMLWLinkDevInfo devInfo = *it;
        if ( deviceId == devInfo.getDeviceId() ) {
            pciInfo = devInfo.getDevicePCIInfo();
            break;
        }
        it++;
    }
    return pciInfo;
}

uint64
LocalFMLWLinkDevRepo::getDeviceId(lwlink_pci_dev_info pciInfo)
{
    FMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();
    uint64 deviceId = -1;

    while ( it != mLWLinkDevList.end() ) {
        FMLWLinkDevInfo devInfo = *it;
        lwlink_pci_dev_info tmpPciInfo = devInfo.getDevicePCIInfo();
        // implement operator== instead
        if ((tmpPciInfo.domain == pciInfo.domain) && (tmpPciInfo.bus == pciInfo.bus) &&
             (tmpPciInfo.device == pciInfo.device) && (tmpPciInfo.function == pciInfo.function)) {
            // found the device. get device id information
            deviceId = devInfo.getDeviceId();
            break;
        }
        it++;
    }

    return deviceId;
}

void
LocalFMLWLinkDevRepo::setDevLinkInitStatus(uint64 deviceId,
                                           FMLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN])
{
    FMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();

    while ( it != mLWLinkDevList.end() ) {
        FMLWLinkDevInfo &devInfo = *it;
        if ( deviceId == devInfo.getDeviceId() ) {
            devInfo.setLinkInitStatus( initStatus );
            break;
        }
        it++;
    }
}

void
LocalFMLWLinkDevRepo::setLocalNodeId(uint32 nodeId)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    // this will set the node id to the lwlink driver
    // after this, the device information is populated again
    // in LFM context.
    mLocalNodeId = nodeId;
    lwlink_set_node_id nodeIdParam;
    nodeIdParam.nodeId = mLocalNodeId;
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_NODE_ID, &nodeIdParam, sizeof(nodeIdParam) );
#endif // LW_MODS_GDM_BUILD

    //
    // Ideally, we should be fetching LWLink device information when LFM
    // starts. Since LFM don't have a persistent storage, it must
    // wait for the GFM to tell its local fabric ID. Additionally, GFM
    // will disable certain links depending on the configuration it is
    // activating (like multi-host with trunk links disabled). So wait
    // for all those config to finish before fetching LWLink device
    // information from LWLinkCoreLib driver.
    //
}

void
LocalFMLWLinkDevRepo::populateLWLinkDeviceList(void)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    lwlink_get_devices_info getParam;
    memset( &getParam, 0, sizeof(getParam) );

    // clean any outstanding device information
    mLWLinkDevList.clear();

    // query the driver for device information
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_GET_DEVICES_INFO, &getParam, sizeof(getParam) );
    if ( getParam.status != LWL_SUCCESS ) {
        return;
    }

    for ( uint32 numDevice = 0; numDevice < getParam.numDevice; numDevice++ ) {
        FMLWLinkDevInfo devInfo( mLocalNodeId, getParam.devInfo[numDevice] );
        uint64 deviceId = devInfo.getDeviceId();
        // update our local device information
        mLWLinkDevList.push_back(devInfo);
    }
#endif // LW_MODS_GDM_BUILD
}

void
LocalFMLWLinkDevRepo::dumpInfo(std::ostream *os)
{
    FMLWLinkDevInfoList::iterator it;

    for ( it = mLWLinkDevList.begin(); it != mLWLinkDevList.end(); it++ ) {
        FMLWLinkDevInfo devInfo = (*it);
        devInfo.dumpInfo( os );
    }
}

GlobalFMLWLinkDevRepo::GlobalFMLWLinkDevRepo()
{
    mLWLinkDevPerNode.clear();
}

GlobalFMLWLinkDevRepo::~GlobalFMLWLinkDevRepo()
{
    mLWLinkDevPerNode.clear();
}

void
GlobalFMLWLinkDevRepo::addDeviceInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoRsp &devInfoRsp)
{
    int idx = 0;
    FMLWLinkDevPerId devPerId;

    // erase existing (if any) device information of the node and re-add
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it != mLWLinkDevPerNode.end() ) {
        mLWLinkDevPerNode.erase(it);
    }

    // parse the node's LWLink device information received from LFM
    for ( idx = 0; idx < devInfoRsp.devinfo_size(); idx++ ) {
        lwswitch::lwlinkDeviceInfoMsg infoMsg = devInfoRsp.devinfo( idx );
        FMLWLinkDevInfo lwLinkDevInfo( nodeId, infoMsg );
        uint64 devId = lwLinkDevInfo.getDeviceId();
        devPerId.insert( std::make_pair(devId, lwLinkDevInfo) );
    }

    // add node's device information locally
    mLWLinkDevPerNode.insert( std::make_pair(nodeId, devPerId) );
}

bool
GlobalFMLWLinkDevRepo::getDeviceInfo(uint32 nodeId, uint64 devId, FMLWLinkDevInfo &devInfo)
{
    // first get the corresponding node and then the device using devId
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it != mLWLinkDevPerNode.end() ) {
        FMLWLinkDevPerId devPerId = it->second;
        FMLWLinkDevPerId::iterator jit = devPerId.find( devId );
        if ( jit != devPerId.end() ) {
            // found the device
            devInfo = jit->second;
            return true;
        }
    }
    return false;
}

bool
GlobalFMLWLinkDevRepo::getDeviceInfo(uint32 nodeId, FMPciInfo_t pciInfo, FMLWLinkDevInfo &devInfo)
{
    // first get the corresponding node
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    // search all the devices in the specified node.
    FMLWLinkDevPerId devPerId = it->second;
    FMLWLinkDevPerId::iterator jit;
    for ( jit = devPerId.begin(); jit != devPerId.end(); jit++) {
        FMLWLinkDevInfo tempDevInfo = jit->second;
        lwlink_pci_dev_info lwlinkPciInfo = tempDevInfo.getDevicePCIInfo();
        if ((pciInfo.domain == lwlinkPciInfo.domain) &&
             (pciInfo.bus == lwlinkPciInfo.bus) &&
             (pciInfo.device == lwlinkPciInfo.device) &&
             (pciInfo.function == lwlinkPciInfo.function)) {
            // found the desired device
            devInfo = tempDevInfo;
            return true;
        }
    }

    // not found the specified device.
    return false;
}

bool
GlobalFMLWLinkDevRepo::setDeviceLinkInitStatus(uint32 nodeId, FMLinkInitStatusInfoList &statusInfo)
{
    // first get the corresponding node and update device init status
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    // update all the device's init status
    FMLinkInitStatusInfoList::iterator jit;
    FMLWLinkDevPerId &devPerId = it->second;
    for ( jit = statusInfo.begin(); jit != statusInfo.end(); jit++) {
        FMLinkInitStatusInfo initInfo = (*jit);
        FMLWLinkDevPerId::iterator devIt = devPerId.find( initInfo.gpuOrSwitchId );
            if ( devIt != devPerId.end() ) {
                // found the device
                FMLWLinkDevInfo &devInfo = devIt->second;
                devInfo.setLinkInitStatus( initInfo.initStatus );
            }
    }

    return true;
}

bool
GlobalFMLWLinkDevRepo::setDeviceLinkState(uint32 nodeId, uint64 devId, uint32 linkIndex,
                                          FMLWLinkStateInfo linkState)
{
    // first get the corresponding node
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    FMLWLinkDevPerId &devPerId = it->second;
    FMLWLinkDevPerId::iterator jit = devPerId.find( devId );
    if ( jit == devPerId.end() ) {
        // not found the specified device
        return false;
    }

    // found the device, update it's link state.
    FMLWLinkDevInfo &devInfo = jit->second;
    return devInfo.setLinkState( linkIndex, linkState );
}

bool
GlobalFMLWLinkDevRepo::getDeviceList(uint32 nodeId, FMLWLinkDevInfoList &devList)
{
    devList.clear();
    // first get the corresponding node
    FMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    FMLWLinkDevPerId devPerId = it->second;
    FMLWLinkDevPerId::iterator jit;
    for ( jit = devPerId.begin(); jit != devPerId.end(); jit++) {
        devList.push_back( (jit->second) );
    }

    return true;
}

bool
GlobalFMLWLinkDevRepo::mergeDeviceInfoForNode(uint32 nodeId, GlobalFMLWLinkDevRepo &srclwLinkDevRepo)
{
    //
    // for the given node, add LWLink devices which are present in srclwLinkDevRepo but not present
    // in this class's device repo
    //

    FMLWLinkDevInfoPerNode &srcDevInfoPerNode = srclwLinkDevRepo.mLWLinkDevPerNode;
    FMLWLinkDevInfoPerNode::iterator srcIt = srcDevInfoPerNode.find( nodeId );
    if ( srcIt == srcDevInfoPerNode.end() ) {
        // no devices found for the specified node. nothing to add
        return false;
    }

    // check whether the destination has devices for this node
    if ( mLWLinkDevPerNode.find(nodeId) == mLWLinkDevPerNode.end() ) {
        //
        // no such node exists for us. technically this shouldn't happen even if all the GPUs
        // have LWLinks disabled. We should have at least LWSwitches as LWLink devices here
        //
        return false;
    }

    // get the destination device list map for the specified node
    FMLWLinkDevPerId &dstDevPerIdMap = mLWLinkDevPerNode[nodeId];

    // go through each device and add missing devices
    FMLWLinkDevPerId srcDevPerId = srcIt->second;
    FMLWLinkDevPerId::iterator it;
    for ( it = srcDevPerId.begin(); it != srcDevPerId.end(); it++ ) {
        uint64 srcDevId = it->first;
        FMLWLinkDevInfo srcLWLinkDevInfo = it->second;
        if ( dstDevPerIdMap.find(srcDevId) == dstDevPerIdMap.end() ) {
            // this device id is not present in destination, add it
            dstDevPerIdMap.insert( std::make_pair(srcDevId, srcLWLinkDevInfo) );
        }
    }

    return true;
}

void
GlobalFMLWLinkDevRepo::dumpInfo(std::ostream *os)
{
    FMLWLinkDevInfoPerNode::iterator it;
    for ( it = mLWLinkDevPerNode.begin(); it != mLWLinkDevPerNode.end(); it++ ) {
        *os << "\t Dumping information for Node Index " << int(it->first) << std::endl;
        FMLWLinkDevPerId devPerId = it->second;
        FMLWLinkDevPerId::iterator jit;
        for ( jit = devPerId.begin(); jit != devPerId.end(); jit++ ) {
            FMLWLinkDevInfo devInfo = jit->second;
            devInfo.dumpInfo( os );
        }
    }
}

FMPciInfo_t
FMLWLinkDevInfo::getPciInfo()
{
    FMPciInfo_t pciInfo;

    pciInfo.domain = mPciInfo.domain;
    pciInfo.bus = mPciInfo.bus;
    pciInfo.device = mPciInfo.device;
    pciInfo.function = mPciInfo.function; //RM don't provide this information.
    snprintf(pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
             FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&pciInfo));

    return pciInfo;
}

