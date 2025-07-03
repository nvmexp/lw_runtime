#include <string.h>

#include "logging.h"
#include "dcgm_structs.h"
#include "lwlink_errors.h"
#include "DcgmFMLWLinkDeviceRepo.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFMTopologyValidator.h"

DcgmFMLWLinkDevInfo::DcgmFMLWLinkDevInfo()
{
    mDeviceId = 0;
    memset(mDevUuid, 0, LWLINK_UUID_LEN);
    mNumLinks = 0;
    mDevType = 0;
    mEnabledLinkMask = 0;
    mNodeId = 0;

    mLinkInitStatus.clear();
    mLinkStateInfo.clear();
}

DcgmFMLWLinkDevInfo::DcgmFMLWLinkDevInfo(uint32 nodeId, lwlink_detailed_dev_info devInfo)
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
}

DcgmFMLWLinkDevInfo::DcgmFMLWLinkDevInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoMsg &devInfoMsg)
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
}

void
DcgmFMLWLinkDevInfo::setLinkInitStatus(DcgmLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN])
{
    for( uint32 idx = 0; idx < LWLINK_MAX_DEVICE_CONN; idx++ ) {
        DcgmLinkInitInfo tempStatus;
        tempStatus.initStatus = initStatus[idx].initStatus;
        tempStatus.linkIndex = initStatus[idx].linkIndex;
        mLinkInitStatus.push_back( tempStatus );
    }
}

bool
DcgmFMLWLinkDevInfo::setLinkState(uint32 linkIndex, DcgmLWLinkStateInfo linkState)
{
    if (linkIndex >= LWLINK_MAX_DEVICE_CONN) {
        return false;
    }

    mLinkStateInfo.insert( std::make_pair(linkIndex, linkState) );
    return true;
}

void
DcgmFMLWLinkDevInfo::getInitFailedLinksIndex(std::list<uint32> &initFailedLinks)
{
    // Link initialization status from LWLinkCoreLib driver is conselwtive. So
    // we need a contiguous array index while referencing it as some links (linkIdx)
    // may not be enabled.
    uint32 initIdx = 0;

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) )
            continue;

        DcgmLinkInitInfo tempStatus = mLinkInitStatus[initIdx];
        if ( tempStatus.initStatus == false ) {
            initFailedLinks.push_back(tempStatus.linkIndex);
        }
        initIdx++;
    }
}

bool
DcgmFMLWLinkDevInfo::isAllLinksInitialized(void)
{
    // Link initialization status from LWLinkCoreLib driver is conselwtive. So
    // we need a contiguous array index while referencing it as some links (linkIdx)
    // may not be enabled.
    uint32 initIdx = 0;

    for ( uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++ ) {
        // skip if the link is not enabled
        if ( !(mEnabledLinkMask & ((uint64)1 << linkIdx)) )
            continue;

        DcgmLinkInitInfo tempStatus = mLinkInitStatus[initIdx];
        if ( tempStatus.initStatus == false ) {
            // not all the links are initialized
            return false;
        }
        initIdx++;
    }
    return true;
}

bool
DcgmFMLWLinkDevInfo::publishLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm)
{
    if ( mDevType == lwlink_device_type_gpu ) {
        // publish GPU link state to DCGM cache manager
        return publishGpuLinkStateToCacheManager( pGfm );
    } else if ( mDevType == lwlink_device_type_lwswitch ) {
        // publish LWSwitch link state to DCGM cache manager
        return publishLWSwitchLinkStateToCacheManager( pGfm );
    } else {
        PRINT_ERROR("", "unsupported device type while publishing link state to DCGM cache manager");
        return false;
    }

    return true;
}

bool
DcgmFMLWLinkDevInfo::publishGpuLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm)
{
    dcgmReturn_t dcgmRetVal;
    uint32_t gpuId;

    if ( !getGpuEnumIndex( pGfm, gpuId ) ) {
        PRINT_ERROR("%lld", "getGpuEnumIndex failed for device id %lld", mDeviceId);
        return false;
    }

    for( uint32 idx = 0; idx < DCGM_LWLINK_MAX_LINKS_PER_GPU; idx++ ) {
        dcgmLwLinkLinkState_t dcgmLinkState;

        // all the enabled GPU links are supposed to be connected for bare metal
        // and VM configuration. So skipping cross checking with topology information.
        if ( !(mEnabledLinkMask & ((uint64)1 << idx)) ) {
            // this link is disabled in RM.
            dcgmLinkState = DcgmLwLinkLinkStateDisabled;
        } else {
            DcgmPerLinkStateInfo::iterator it = mLinkStateInfo.find( idx );
            if ( it == mLinkStateInfo.end() ) {
                PRINT_ERROR("%d", "GPU Link state information is missing for link index %d", idx);
                // treat the link as disabled
                dcgmLinkState = DcgmLwLinkLinkStateDisabled;
            } else {
                // found link state for the specified idx, translate state to DCGM specific value.
                DcgmLWLinkStateInfo linkState = it->second;
                if ( linkState.linkMode == lwlink_link_mode_active ) {
                    dcgmLinkState = DcgmLwLinkLinkStateUp;
                } else {
                    dcgmLinkState = DcgmLwLinkLinkStateDown;
                }
            }
        }

        // publish the link state to cache manager
        DcgmCacheManager *pCacheManager = pGfm->GetCacheManager();
        dcgmRetVal = pCacheManager->SetEntityLwLinkLinkState( DCGM_FE_GPU, gpuId, idx, dcgmLinkState );
        if (DCGM_ST_OK != dcgmRetVal) {
            PRINT_ERROR("%d %d %d", "SetEntityLwLinkLinkState failed for GPU index %d link index %d with error %d",
                        gpuId, idx, dcgmRetVal);
            return false;
        }
    }

    return true;
}

bool
DcgmFMLWLinkDevInfo::publishLWSwitchLinkStateToCacheManager(DcgmGlobalFabricManager *pGfm)
{
    dcgmReturn_t dcgmRetVal;
    uint32_t physicalId;

    if ( !getLWSwitchPhysicalId( pGfm, physicalId ) ) {
        PRINT_ERROR("%lld", "getLWSwitchPhysicalId failed for device id %lld", mDeviceId);
        return false;
    }

    for( uint32 idx = 0; idx < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; idx++ ) {
        dcgmLwLinkLinkState_t dcgmLinkState;
        if ( !(mEnabledLinkMask & ((uint64)1 << idx)) ) {
            // this link is disabled in LWSwitch driver.
            dcgmLinkState = DcgmLwLinkLinkStateDisabled;
        } else {
            DcgmPerLinkStateInfo::iterator it = mLinkStateInfo.find( idx );
            if ( it == mLinkStateInfo.end() ) {
                // some LWSwitch ports may not be physically wired and cannot be disabled in driver
                // (due to sharing of clocks). We will not have corresponding port's link state 
                // updated in our context as there will not be any LWLink connection for those ports.
                // Check our topology for such unused ports and set DCGM link state appropriately.
                if (!pGfm->mTopoValidator->isSwitchPortConnected( mNodeId, physicalId, idx )) {
                    // this is expected, so no need to log error.
                    dcgmLinkState = DcgmLwLinkLinkStateDisabled;
                } else {
                    PRINT_ERROR("%d", "LWSwitch Link state information is missing for port index %d", idx);
                    // treat the link as disabled
                    dcgmLinkState = DcgmLwLinkLinkStateDisabled;
                }
            } else {
                // found link state for the port idx, translate state to DCGM specific value.
                DcgmLWLinkStateInfo linkState = it->second;
                if ( linkState.linkMode == lwlink_link_mode_active ) {
                    dcgmLinkState = DcgmLwLinkLinkStateUp;
                } else {
                    dcgmLinkState = DcgmLwLinkLinkStateDown;
                }
           }
        }

        // publish the link state to cache manager
        DcgmCacheManager *pCacheManager = pGfm->GetCacheManager();
        dcgmRetVal = pCacheManager->SetEntityLwLinkLinkState( DCGM_FE_SWITCH, physicalId, idx, dcgmLinkState );
        if (DCGM_ST_OK != dcgmRetVal) {
            PRINT_ERROR("%d %d %d", "SetEntityLwLinkLinkState failed for LWSwitch physical id %d port %d with error %d",
                        physicalId, idx, dcgmRetVal);
            return false;
        }
    }

    return true;
}

bool
DcgmFMLWLinkDevInfo::getGpuEnumIndex(DcgmGlobalFabricManager *pGfm, uint32_t &gpuEnumIdx)
{
    DcgmFMGpuInfoMap gpuInfoMap = pGfm->getGpuInfoMap();
    DcgmFMGpuInfoMap::iterator it = gpuInfoMap.find( mNodeId );
    if ( it == gpuInfoMap.end() ) {
        return false;
    }

    DcgmFMGpuInfoList gpuList = it->second;
    DcgmFMGpuInfoList::iterator jit;
    for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
        DcgmFMGpuInfo tempInfo = (*jit);
        if ( (tempInfo.pciInfo.domain == mPciInfo.domain) &&
             (tempInfo.pciInfo.bus == mPciInfo.bus) &&
             (tempInfo.pciInfo.device == mPciInfo.device) &&
             (tempInfo.pciInfo.function == mPciInfo.function) ) {
            gpuEnumIdx = tempInfo.gpuIndex;
            return true;
        }
    }

    return false;
}


bool
DcgmFMLWLinkDevInfo::getLWSwitchPhysicalId(DcgmGlobalFabricManager *pGfm, uint32_t &physicalId)
{
    DcgmFMLWSwitchInfoMap switchInfoMap = pGfm->getLwSwitchInfoMap();
    DcgmFMLWSwitchInfoMap::iterator it = switchInfoMap.find( mNodeId );
    if ( it == switchInfoMap.end() ) {
        return false;
    }

    DcgmFMLWSwitchInfoList switchList = it->second;
    DcgmFMLWSwitchInfoList::iterator jit;
    for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
        DcgmFMLWSwitchInfo tempInfo = (*jit);
        if ( (tempInfo.pciInfo.domain == mPciInfo.domain) &&
             (tempInfo.pciInfo.bus == mPciInfo.bus) &&
             (tempInfo.pciInfo.device == mPciInfo.device) &&
             (tempInfo.pciInfo.function == mPciInfo.function) ) {
            physicalId = tempInfo.physicalId;
            return true;
        }
    }

    return false;
}

void
DcgmFMLWLinkDevInfo::dumpInfo(std::ostream *os)
{
    *os << "\t Device Name:" << mDeviceName << std::endl;
    *os << "\t PCI Info:" << std::endl;
    *os << "\t\t Domain:" << std::hex << int(mPciInfo.domain) << std::endl;
    *os << "\t\t Bus:" << std::hex << int(mPciInfo.bus) << std::endl;
    *os << "\t\t Device:" << std::hex << int(mPciInfo.device) << std::endl;
    *os << "\t\t Function:" << std::hex << int(mPciInfo.function) << std::endl;
    *os << "\t Device Type:" << std::dec << mDevType << std::endl;
    *os << "\t Number of Links:" << std::dec <<  mNumLinks << std::endl;
    *os << "\t Link Initialization Status" << std::endl;
    for ( uint32 idx = 0; idx < mNumLinks; idx++) {
        DcgmLinkInitInfo tempStatus = mLinkInitStatus[idx];
        if ( tempStatus.initStatus ) {
            *os << "\t\t\t Link Index " << std::dec << tempStatus.linkIndex << ":Success" << std::endl;
        } else {
            *os << "\t\t\t Link Index " << std::dec << tempStatus.linkIndex << ":Failed" << std::endl;
        }
    }
}

DcgmFMLWLinkDevInfo::~DcgmFMLWLinkDevInfo()
{
    // nothing specific
}

DcgmLFMLWLinkDevRepo::DcgmLFMLWLinkDevRepo(DcgmFMLWLinkDrvIntf *linkDrvIntf)
{
    // query the lwlink driver and populate all the registered device information
    mLWLinkDevList.clear();
    mLWLinkDrvIntf = linkDrvIntf;
    mLocalNodeId = 0;
}

DcgmLFMLWLinkDevRepo::~DcgmLFMLWLinkDevRepo()
{
    // update our local device information
    mLWLinkDevList.clear();
}

lwlink_pci_dev_info
DcgmLFMLWLinkDevRepo::getDevicePCIInfo(uint64 deviceId)
{
    // return empty values if not found
    lwlink_pci_dev_info pciInfo = {0};
    DcgmFMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();

    while ( it != mLWLinkDevList.end() ) {
        DcgmFMLWLinkDevInfo devInfo = *it;
        if ( deviceId == devInfo.getDeviceId() ) {
            pciInfo = devInfo.getDevicePCIInfo();
            break;
        }
        it++;
    }
    return pciInfo;
}

uint64
DcgmLFMLWLinkDevRepo::getDeviceId(lwlink_pci_dev_info pciInfo)
{
    DcgmFMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();
    uint64 deviceId = -1;

    while ( it != mLWLinkDevList.end() ) {
        DcgmFMLWLinkDevInfo devInfo = *it;
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
DcgmLFMLWLinkDevRepo::setDevLinkInitStatus(uint64 deviceId,
                                           DcgmLinkInitInfo initStatus[LWLINK_MAX_DEVICE_CONN])
{
    DcgmFMLWLinkDevInfoList::iterator it = mLWLinkDevList.begin();

    while ( it != mLWLinkDevList.end() ) {
        DcgmFMLWLinkDevInfo &devInfo = *it;
        if ( deviceId == devInfo.getDeviceId() ) {
            devInfo.setLinkInitStatus( initStatus );
            break;
        }
        it++;
    }
}

void
DcgmLFMLWLinkDevRepo::setLocalNodeId(uint32 nodeId)
{
    // this will set the node id to the lwlink driver
    // after this, the device information is populated again
    // in LFM context.
    mLocalNodeId = nodeId;
    lwlink_set_node_id nodeIdParam;
    nodeIdParam.nodeId = mLocalNodeId;
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_NODE_ID, &nodeIdParam );

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
DcgmLFMLWLinkDevRepo::populateLWLinkDeviceList(void)
{
    lwlink_get_devices_info getParam;
    memset( &getParam, 0, sizeof(getParam) );

    // clean any outstanding device information
    mLWLinkDevList.clear();

    // query the driver for device information
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_GET_DEVICES_INFO, &getParam );
    if ( getParam.status != LWL_SUCCESS ) {
        return;
    }

    for ( uint32 numDevice = 0; numDevice < getParam.numDevice; numDevice++ ) {
        DcgmFMLWLinkDevInfo devInfo( mLocalNodeId, getParam.devInfo[numDevice] );
        uint64 deviceId = devInfo.getDeviceId();
        // update our local device information
        mLWLinkDevList.push_back(devInfo);
    }
}

void
DcgmLFMLWLinkDevRepo::dumpInfo(std::ostream *os)
{
    DcgmFMLWLinkDevInfoList::iterator it;

    for ( it = mLWLinkDevList.begin(); it != mLWLinkDevList.end(); it++ ) {
        DcgmFMLWLinkDevInfo devInfo = (*it);
        devInfo.dumpInfo( os );
    }
}

DcgmGFMLWLinkDevRepo::DcgmGFMLWLinkDevRepo()
{
    mLWLinkDevPerNode.clear();
}

DcgmGFMLWLinkDevRepo::~DcgmGFMLWLinkDevRepo()
{
    mLWLinkDevPerNode.clear();
}

void
DcgmGFMLWLinkDevRepo::addDeviceInfo(uint32 nodeId, lwswitch::lwlinkDeviceInfoRsp &devInfoRsp)
{
    int idx = 0;
    DcgmFMLWLinkDevPerId devPerId;

    // erase existing (if any) device information of the node and re-add
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it != mLWLinkDevPerNode.end() ) {
        mLWLinkDevPerNode.erase(it);
    }

    // parse the node's LWLink device information received from LFM
    for ( idx = 0; idx < devInfoRsp.devinfo_size(); idx++ ) {
        lwswitch::lwlinkDeviceInfoMsg infoMsg = devInfoRsp.devinfo( idx );
        DcgmFMLWLinkDevInfo lwLinkDevInfo( nodeId, infoMsg );
        uint64 devId = lwLinkDevInfo.getDeviceId();
        devPerId.insert( std::make_pair(devId, lwLinkDevInfo) );
    }

    // add node's device information locally
    mLWLinkDevPerNode.insert( std::make_pair(nodeId, devPerId) );
}

bool
DcgmGFMLWLinkDevRepo::getDeviceInfo(uint32 nodeId, uint64 devId, DcgmFMLWLinkDevInfo &devInfo)
{
    // first get the corresponding node and then the device using devId
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it != mLWLinkDevPerNode.end() ) {
        DcgmFMLWLinkDevPerId devPerId = it->second;
        DcgmFMLWLinkDevPerId::iterator jit = devPerId.find( devId );
        if ( jit != devPerId.end() ) {
            // found the device
            devInfo = jit->second;
            return true;
        }
    }
    return false;
}

bool
DcgmGFMLWLinkDevRepo::getDeviceInfo(uint32 nodeId, DcgmFMPciInfo pciInfo, DcgmFMLWLinkDevInfo &devInfo)
{
    // first get the corresponding node
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    // search all the devices in the specified node.
    DcgmFMLWLinkDevPerId devPerId = it->second;
    DcgmFMLWLinkDevPerId::iterator jit;
    for ( jit = devPerId.begin(); jit != devPerId.end(); jit++) {
        DcgmFMLWLinkDevInfo tempDevInfo = jit->second;
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
DcgmGFMLWLinkDevRepo::setDeviceLinkInitStatus(uint32 nodeId, DcgmLinkInitStatusInfoList &statusInfo)
{
    // first get the corresponding node and update device init status
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    // update all the device's init status
    DcgmLinkInitStatusInfoList::iterator jit;
    DcgmFMLWLinkDevPerId &devPerId = it->second;
    for ( jit = statusInfo.begin(); jit != statusInfo.end(); jit++) {
        DcgmLinkInitStatusInfo initInfo = (*jit);
        DcgmFMLWLinkDevPerId::iterator devIt = devPerId.find( initInfo.gpuOrSwitchId );
            if ( devIt != devPerId.end() ) {
                // found the device
                DcgmFMLWLinkDevInfo &devInfo = devIt->second;
                devInfo.setLinkInitStatus( initInfo.initStatus );
            }
    }

    return true;
}

bool
DcgmGFMLWLinkDevRepo::setDeviceLinkState(uint32 nodeId, uint64 devId, uint32 linkIndex,
                                                DcgmLWLinkStateInfo linkState)
{
    // first get the corresponding node
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    DcgmFMLWLinkDevPerId &devPerId = it->second;
    DcgmFMLWLinkDevPerId::iterator jit = devPerId.find( devId );
    if ( jit == devPerId.end() ) {
        // not found the specified device
        return false;
    }

    // found the device, update it's link state.
    DcgmFMLWLinkDevInfo &devInfo = jit->second;
    return devInfo.setLinkState( linkIndex, linkState );
}

bool
DcgmGFMLWLinkDevRepo::publishNodeLinkStateToCacheManager(uint32 nodeId, DcgmGlobalFabricManager *pGfm)
{
    bool retVal = false;
    // first get the corresponding node
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    // update link state for all the devices in this node
    DcgmFMLWLinkDevPerId devPerId = it->second;
    DcgmFMLWLinkDevPerId::iterator jit;
    for ( jit = devPerId.begin(); jit != devPerId.end(); jit++) {
        DcgmFMLWLinkDevInfo devInfo = jit->second;
        retVal = devInfo.publishLinkStateToCacheManager( pGfm );
        if ( !retVal ) {
            // error while publishing link state. error is already logged
            // by helper functions
            break;
        }
    }

    return retVal;
}

bool
DcgmGFMLWLinkDevRepo::getDeviceList(uint32 nodeId, DcgmFMLWLinkDevInfoList &devList)
{
    devList.clear();
    // first get the corresponding node
    DcgmFMLWLinkDevInfoPerNode::iterator it = mLWLinkDevPerNode.find( nodeId );
    if ( it == mLWLinkDevPerNode.end() ) {
        // no such node exists for us
        return false;
    }

    DcgmFMLWLinkDevPerId devPerId = it->second;
    DcgmFMLWLinkDevPerId::iterator jit;
    for ( jit = devPerId.begin(); jit != devPerId.end(); jit++) {
        devList.push_back( (jit->second) );
    }

    return true;
}

void
DcgmGFMLWLinkDevRepo::dumpInfo(std::ostream *os)
{
    DcgmFMLWLinkDevInfoPerNode::iterator it;
    for ( it = mLWLinkDevPerNode.begin(); it != mLWLinkDevPerNode.end(); it++ ) {
        *os << "\t Dumping information for Node Index " << int(it->first) << std::endl;        
        DcgmFMLWLinkDevPerId devPerId = it->second;
        DcgmFMLWLinkDevPerId::iterator jit;
        for ( jit = devPerId.begin(); jit != devPerId.end(); jit++ ) {
            DcgmFMLWLinkDevInfo devInfo = jit->second;
            devInfo.dumpInfo( os );
        }
    }
}

