#include <iostream>
#include <sstream>

#include "DcgmModuleLwSwitch.h"
#include "logging.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFMTopologyValidator.h"
#include "DcgmLogging.h"

DcgmFMTopologyValidator::DcgmFMTopologyValidator(DcgmGlobalFabricManager *pGfm)
{
    mGfm = pGfm;
};

DcgmFMTopologyValidator::~DcgmFMTopologyValidator()
{
    // nothing as of now
};

int
DcgmFMTopologyValidator::validateTopologyInfo(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = true;

    if (!mGfm->mpParser->isSwtichGpioPresent()) {
        // we can't validate if there is no switch GPIO based id is specified in
        // topology. return success to keep FM running as this may not be a
        // production system
        return true;
    }

    // validate number of LWSwitches and trunk connections.
    // access connections are not validated as gpu/access link failures
    // will manifest as gpu degraded mode.

    // gpu count validation is not strictly enforced as some gpus may be blacklisted
    // as part of degraded mode and RM may not always return correct number of
    // blacklisted gpu count (such gpu pci enumeration itself fails)

    for (nodeit = mGfm->mpParser->NodeCfg.begin(); nodeit!= mGfm->mpParser->NodeCfg.end(); nodeit++) {
        NodeConfig *pNode = nodeit->second;
        uint32_t nodeId = pNode->nodeId;

        DcgmFMGpuInfoList gpuInfo = mGfm->mGpuInfoMap[nodeId];
        uint32_t gpuCount = gpuInfo.size();

        DcgmFMGpuInfoList blacklistGpuInfo = mGfm->mBlacklistGpuInfoMap[nodeId];
        uint32_t blacklistGpuCount = blacklistGpuInfo.size();

        DcgmFMLWSwitchInfoList switchInfo = mGfm->mLwswitchInfoMap[nodeId];
        uint32_t switchCount = switchInfo.size();


        // count all the intra node trunk connections discovered
        uint32_t lwLinkIntraTrunkConnCount = getIntraNodeTrunkConnCount(nodeId, mGfm->mLWLinkConnRepo);

        // count all the inter node trunk connections discovered
        uint32_t lwLinkInterTrunkConnCount = getInterNodeTrunkConnCount(nodeId);

        if (!validateNodeTopologyInfo(pNode, switchCount, lwLinkIntraTrunkConnCount,
                                      lwLinkInterTrunkConnCount, gpuCount, blacklistGpuCount)) {
            PRINT_ERROR("%d %d %d ", "Detected node topology is not matching with system topology information for node: %d intra:%d inter:%d", 
                        nodeId,lwLinkIntraTrunkConnCount, lwLinkInterTrunkConnCount);
            retVal = false; // continue validating other nodes, but return status as failure.
        }
    }

    return retVal;
}

int
DcgmFMTopologyValidator::validateNodeTopologyInfo(NodeConfig *pNodeCfg, uint32_t switchCount,
                                                  uint32_t lwLinkIntraTrunkConnCount, uint32_t lwLinkInterTrunkConnCount,
                                                  uint32_t gpuCount, uint32_t blacklistGpuCount)
{
    int idx = 0;

    // check for bare metal partition
    for (idx = 0; idx < pNodeCfg->partitionInfo.baremetalinfo_size(); idx++) {
        const bareMetalPartitionInfo &bareMetalInfo = pNodeCfg->partitionInfo.baremetalinfo(idx);
        const partitionMetaDataInfo &metaDataInfo = bareMetalInfo.metadata();
        if ((metaDataInfo.switchcount() == switchCount) &&
            (metaDataInfo.lwlinkintratrunkconncount() == lwLinkIntraTrunkConnCount) &&
            (metaDataInfo.lwlinkintertrunkconncount() == lwLinkInterTrunkConnCount)) {
            // matched with bare metal partition
            // gpu count validation is not strictly enforced, log a notice for any mismatch
            if (metaDataInfo.gpucount() != (gpuCount + blacklistGpuCount)) {
                SYSLOG_NOTICE("detected GPU count is not matching with topology");
            }
            return true;
        }
    }

    // check with Pass-through virtualization partition information
    for (idx = 0; idx < pNodeCfg->partitionInfo.ptvirtualinfo_size(); idx++) {
        const ptVMPartitionInfo &ptPartInfo = pNodeCfg->partitionInfo.ptvirtualinfo(idx);
        const partitionMetaDataInfo &metaDataInfo = ptPartInfo.metadata();
        if ((metaDataInfo.switchcount() == switchCount) &&
            (metaDataInfo.lwlinkintratrunkconncount() == lwLinkIntraTrunkConnCount) &&
            (metaDataInfo.lwlinkintertrunkconncount() == lwLinkInterTrunkConnCount)) {
            // matched with a Pass-through virtualization partition
            // gpu count validation is not strictly enforced, log a notice for any mismatch
            if (metaDataInfo.gpucount() != (gpuCount + blacklistGpuCount)) {
                SYSLOG_NOTICE("detected GPU count is not matching with topology");
            }
            return true;
        }
    }

    // no match case
    return false;
}

bool
DcgmFMTopologyValidator::isAllLWLinkTrunkConnsActive(void)
{
    // make sure that all the LWLink trunk connections are at ACTIVE (HIGH SPEED) mode.
    // trunk connection failures are treated as fatal error.

    // for cases like GPU baseboards are connected to different motherboards,
    // trunk connections will not be discovered and will be covered in 
    // total number of access/trunk connection validations.

    // first validate all node's intra node connections
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    for (nodeit = mGfm->mpParser->NodeCfg.begin(); nodeit!= mGfm->mpParser->NodeCfg.end(); nodeit++) {
        NodeConfig *pNode = nodeit->second;
        uint32_t nodeId = pNode->nodeId;
        if (!isIntraNodeTrunkConnsActive(nodeId, mGfm->mLWLinkConnRepo)) {
            // error already logged
            return false;
        }
    }

    // repeat the same for all the detected inter node connections
    if (!isInterNodeTrunkConnsActive()) {
        // error already logged
        return false;
    }

    // all the detected trunk connections are Active
    return true;
}

void
DcgmFMTopologyValidator::checkLWLinkInitStatusForAllDevices(void)
{
    checkLWSwitchLinkInitStatus();
    checkGpuLinkInitStatus();
}

bool
DcgmFMTopologyValidator::mapGpuIndexByLWLinkConns(DcgmFMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
{
    // create the GPU Physical ID to local index (enumeration order) mapping
    // by detected LWLink connections based on the assumption that LWSwitch’s
    // has a physical GPIO based IDs.

    // Note: While creating this mapping, the detected LWLink connections are
    // validated with the connections specified in the topology file. Any
    // mismatch is treated as an error.

    // Note: Only intra-node connections are used for mapping as all the
    // inter-node connections are between switches.

    // clean any existing mapping information
    gpuConnMatrixMap.clear();

    // get the list of discovered LWLink connections
    LWLinkIntraConnMap tempIntraConnMap = mGfm->mLWLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.begin();

    // iterate for all the available nodes and build the mapping
    for (it = tempIntraConnMap.begin(); it != tempIntraConnMap.end(); it++) {
        uint32_t nodeId = it->first;
        DcgmFMGpuLWLinkConnMatrixList connMatrix;
        if (!mapGpuIndexForOneNode(nodeId, connMatrix)) {
            // error already logged
            return false;
        }
        // mapping created for the specified node.
        gpuConnMatrixMap.insert(std::make_pair(nodeId, connMatrix));
    }

    return true;
}

bool
DcgmFMTopologyValidator::mapGpuIndexForOneNode(uint32_t nodeId,
                                               DcgmFMGpuLWLinkConnMatrixList &connMatrix)
{
    // create GPU Physical ID to local index for the specified node.

    LWLinkIntraConnMap tempIntraConnMap = mGfm->mLWLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);
    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator jit;

    // iterate for all the connections in the node.
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        DcgmDetailedEndPointInfo switchEndPointInfo;
        DcgmDetailedEndPointInfo gpuEndPointInfo;

        // no need to create mapping for trunk connections
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }

        // skip if the connection is not trained to active
        if (!lwLinkConn->isConnTrainedToActive()) {
            continue;
        }

        // get the corresponding switch and gpu information from the LWLink Connection
        if (!getAccessConnDetailedEndPointInfo(lwLinkConn, switchEndPointInfo, gpuEndPointInfo)) {
            return false;
        }

        // check whether this connection was specified in topology
        if (!verifyAccessConnection(lwLinkConn)) {
            // error already logged. break from further parsing
            return false;
        }

        // get the corresponding topology connection as specified in the topology file
        TopologyLWLinkConnEndPoint topoSwitchEndPoint;
        TopologyLWLinkConnEndPoint topoGpuEndPoint;        
        topoSwitchEndPoint.nodeId = switchEndPointInfo.nodeId;
        topoSwitchEndPoint.lwswitchOrGpuId = switchEndPointInfo.physicalId;
        topoSwitchEndPoint.portIndex = switchEndPointInfo.linkIndex;
        getTopologyLWLinkConnGpuEndPoint(topoSwitchEndPoint, topoGpuEndPoint);

        // get the corresponding GPUs link mask, this way we will have link mask associated
        // with GPU physical id and enumeration index. (LocalFM reported the mask with enumeration index)
        uint64 gpuEnabledLinkMask;
        if (!getAccessConnGpuLinkMaskInfo(lwLinkConn, gpuEnabledLinkMask)) {
            std::stringstream outStr;
            lwLinkConn->dumpConnInfo(&outStr, mGfm->mLWLinkDevRepo);
            PRINT_ERROR("%s", "Failed to get associated GPU LWLink Mask for Conn: %s", outStr.str().c_str());
            return false;
        }

        // update/populate our GPU connection matrix wit the physical id
        // specified in the topology and what detected by the driver.
        if (!updateGpuConnectionMatrixInfo(connMatrix, gpuEndPointInfo,
             topoGpuEndPoint, gpuEnabledLinkMask)) {
            // error already logged. break from further parsing          
            return false;
        }
    }

    return true;
}

bool
DcgmFMTopologyValidator::updateGpuConnectionMatrixInfo(DcgmFMGpuLWLinkConnMatrixList &connMatrixList,
                                                       DcgmDetailedEndPointInfo gpuEndPointInfo,
                                                       TopologyLWLinkConnEndPoint topoGpuEndPoint,
                                                       uint64 gpuEnabledLinkMask)
{
    DcgmFMGpuLWLinkConnMatrixList::iterator it;
    DcgmFMGpuLWLinkConnMatrix gpuConnMatrix = {0};

    for (it = connMatrixList.begin(); it != connMatrixList.end(); it++) {
        DcgmFMGpuLWLinkConnMatrix tempMatrix = *it;
        if ((tempMatrix.gpuEnumIndex == gpuEndPointInfo.enumIndex) &&
            (tempMatrix.gpuPhyIndex == topoGpuEndPoint.lwswitchOrGpuId)) {
           // found an existing connection matrix
           break;
        }
    }

    if (it != connMatrixList.end()) {
        gpuConnMatrix = (*it);
        connMatrixList.erase(it);
        // check whether this port index has already has a connection
        if (gpuConnMatrix.linkConnStatus[topoGpuEndPoint.portIndex]) {
            // we already have a connection with this port
            // this shouldn't happen
            PRINT_ERROR("%d %d %d", "Duplicate LWLink connection for GPU Physical ID:%d Logical Index:%d for Link:%d\n",
                        topoGpuEndPoint.lwswitchOrGpuId, gpuEndPointInfo.enumIndex, topoGpuEndPoint.portIndex);
            return false;
        }
        // mark this link as connected.
        gpuConnMatrix.linkConnStatus[topoGpuEndPoint.portIndex] = true;
    } else {
        // first connection for this GPU
        gpuConnMatrix.gpuPhyIndex = topoGpuEndPoint.lwswitchOrGpuId;
        gpuConnMatrix.gpuEnumIndex = gpuEndPointInfo.enumIndex;        
        gpuConnMatrix.linkConnStatus[topoGpuEndPoint.portIndex] = true;
        gpuConnMatrix.enabledLinkMask = gpuEnabledLinkMask;
    }

    // re-add/insert the connection info to the list
    connMatrixList.push_back(gpuConnMatrix);

    return true;
}

bool
DcgmFMTopologyValidator::getTopologyLWLinkConnGpuEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                                          TopologyLWLinkConnEndPoint &gpuEndPoint)
{
    TopologyLWLinkConnMap::iterator it = mGfm->mpParser->lwLinkConnMap.find(switchEndPoint.nodeId);

    if (it == mGfm->mpParser->lwLinkConnMap.end()) {
        // no connection for this node
        return false;
    }

    TopologyLWLinkConnList connList = it->second;
    TopologyLWLinkConnList::iterator jit;
    for (jit = connList.begin(); jit != connList.end(); jit++) {
        TopologyLWLinkConn conn = (*jit);
        if ((conn.localEnd.nodeId == switchEndPoint.nodeId) &&
            (conn.localEnd.lwswitchOrGpuId == switchEndPoint.lwswitchOrGpuId) &&
            (conn.localEnd.portIndex == switchEndPoint.portIndex)) {
            gpuEndPoint = conn.farEnd;
            return true;
        }
        // Note: a reverse comparison is not needed as we build the topology connection
        // based on keeping switch as localEnd. If a reverse comparison is required,
        // then we need to explicitly specify the endpoint type as the switch physical ID 
        // can match with GPU physical IDs as well (ie both starts from 0, 1, 2, etc and port also matches)
    }

    // not found any matching topology connection
    return false;
}

bool
DcgmFMTopologyValidator::getTopologyLWLinkConnBySwitchEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                                               TopologyLWLinkConn &topoLWLinkConn)
{
    TopologyLWLinkConnMap::iterator it = mGfm->mpParser->lwLinkConnMap.find(switchEndPoint.nodeId);

    if (it == mGfm->mpParser->lwLinkConnMap.end()) {
        // no connection for this node
        return false;
    }

    TopologyLWLinkConnList connList = it->second;
    TopologyLWLinkConnList::iterator jit;
    for (jit = connList.begin(); jit != connList.end(); jit++) {
        TopologyLWLinkConn conn = (*jit);
        if ((conn.localEnd.nodeId == switchEndPoint.nodeId) &&
            (conn.localEnd.lwswitchOrGpuId == switchEndPoint.lwswitchOrGpuId) &&
            (conn.localEnd.portIndex == switchEndPoint.portIndex)) {
            topoLWLinkConn = conn;
            return true;
        }

        // Note: a reverse comparison is not needed as we build the topology connection
        // based on keeping switch as localEnd. If a reverse comparison is required,
        // then we need to explicitly specify the endpoint type as the switch physical ID 
        // can match with GPU physical IDs as well (ie both starts from 0, 1, 2, etc and port also matches)
    }

    // not found any matching topology connection
    return false;
}

bool
DcgmFMTopologyValidator::isLWLinkTrunkConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn)
{
    DcgmLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    DcgmLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    DcgmFMLWLinkDevInfo end0LWLinkDevInfo; 
    DcgmFMLWLinkDevInfo end1LWLinkDevInfo;

    // get device details as seen by LWLinkCoreLib driver.
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint0.nodeId, endPoint0.gpuOrSwitchId, end0LWLinkDevInfo);
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint1.nodeId, endPoint1.gpuOrSwitchId, end1LWLinkDevInfo);

    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch &&
        end1LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        return true;
    }

    // not a trunk connection
    return false;
}

bool
DcgmFMTopologyValidator::getAccessConnDetailedEndPointInfo(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                                           DcgmDetailedEndPointInfo &switchEndPointInfo,
                                                           DcgmDetailedEndPointInfo &gpuEndPointInfo)
{
    // translate and LWLink connection endpoints seen by LWLinkCoreLib driver
    // to the corresponding devices reported by LWSwtich Driver and RM driver.

    DcgmLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    DcgmLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    DcgmFMLWLinkDevInfo end0LWLinkDevInfo; 
    DcgmFMLWLinkDevInfo end1LWLinkDevInfo;

    // get device details as seen by LWLinkCoreLib driver.
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint0.nodeId, endPoint0.gpuOrSwitchId, end0LWLinkDevInfo);
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint1.nodeId, endPoint1.gpuOrSwitchId, end1LWLinkDevInfo);

    // verify that the connection is access type.
    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch &&
        end1LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        // trunk connection
        return false;
    }

    // access connection. get the corresponding LWSwtich and GPU information using PCI BDF.
    // Note: Assuming current access connections are between GPU and LWswtich (ie no NPU)
    DcgmFMLWSwitchInfo switchInfo;
    DcgmFMGpuInfo gpuInfo;
    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        // assign switch endpoint details accordingly
        if (!getLWSwitchInfoByLWLinkDevInfo(endPoint0.nodeId, end0LWLinkDevInfo, switchInfo)) {
            lwlink_pci_dev_info pciInfo = end0LWLinkDevInfo.getDevicePCIInfo();
            PRINT_ERROR("%x,%x,%x,%x", "No matching LWSwitch device based on LWLinkCoreLib driver PCI BDF %x:%x:%x:%x",
                        pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        // use physicalId or SwtichIndex for comparision based on
        // whether one is specified in topology to support all the kind of systems.
        // Remove this once all the existing topology files are migrated to have
        // switch physical ids.
        if (mGfm->mpParser->isSwtichGpioPresent()) {
            switchEndPointInfo.physicalId = switchInfo.physicalId;
        }else {
            switchEndPointInfo.physicalId = switchInfo.switchIndex;
        }
        switchEndPointInfo.nodeId = endPoint0.nodeId;
        switchEndPointInfo.linkIndex = endPoint0.linkIndex;
        // assign GPU endpoint details accordingly
        if (!getGpuInfoByLWLinkDevInfo(endPoint1.nodeId, end1LWLinkDevInfo, gpuInfo)) {
            lwlink_pci_dev_info pciInfo = end1LWLinkDevInfo.getDevicePCIInfo();
            PRINT_ERROR("%x,%x,%x,%x", "No matching GPU device based on LWLinkCoreLib driver PCI BDF %x:%x:%x:%x",
                        pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        gpuEndPointInfo.enumIndex = gpuInfo.gpuIndex;
        gpuEndPointInfo.nodeId = endPoint1.nodeId;
        gpuEndPointInfo.linkIndex = endPoint1.linkIndex;
    } else {
        if (!getLWSwitchInfoByLWLinkDevInfo(endPoint1.nodeId, end1LWLinkDevInfo, switchInfo)) {
            lwlink_pci_dev_info pciInfo = end1LWLinkDevInfo.getDevicePCIInfo();
            PRINT_ERROR("%x,%x,%x,%x", "No matching LWSwitch device based on LWLinkCoreLib driver PCI BDF %x:%x:%x:%x",
                         pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        // assign switch endpoint details accordingly
        if (mGfm->mpParser->isSwtichGpioPresent()) {
            switchEndPointInfo.physicalId = switchInfo.physicalId;
        }else {
            switchEndPointInfo.physicalId = switchInfo.switchIndex;
        }
        switchEndPointInfo.nodeId = endPoint1.nodeId;
        switchEndPointInfo.linkIndex = endPoint1.linkIndex;
        // assign GPU endpoint details accordingly
        if (!getGpuInfoByLWLinkDevInfo(endPoint0.nodeId, end0LWLinkDevInfo, gpuInfo)) {
            lwlink_pci_dev_info pciInfo = end0LWLinkDevInfo.getDevicePCIInfo();
            PRINT_ERROR("%x,%x,%x,%x", "No matching GPU device based on LWLinkCoreLib driver PCI BDF %x:%x:%x:%x",
                         pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        gpuEndPointInfo.enumIndex = gpuInfo.gpuIndex;
        gpuEndPointInfo.nodeId = endPoint0.nodeId;
        gpuEndPointInfo.linkIndex = endPoint0.linkIndex;
    }

    return true;
}

bool
DcgmFMTopologyValidator::verifyAccessConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn)
{
    // verify whether this access connection is specified in our topology
    DcgmDetailedEndPointInfo switchEndPointInfo;
    DcgmDetailedEndPointInfo gpuEndPointInfo;

    getAccessConnDetailedEndPointInfo(lwLinkConn, switchEndPointInfo, gpuEndPointInfo);

    TopologyLWLinkConnEndPoint topoSwitchEndPoint;
    TopologyLWLinkConnEndPoint topoGpuEndPoint;        
    topoSwitchEndPoint.nodeId = switchEndPointInfo.nodeId;
    topoSwitchEndPoint.lwswitchOrGpuId = switchEndPointInfo.physicalId;
    topoSwitchEndPoint.portIndex = switchEndPointInfo.linkIndex;
    // to verify the connection, check whether we have a remote connection with
    // the specified switch:port exists in the topology.
    if (!getTopologyLWLinkConnGpuEndPoint(topoSwitchEndPoint, topoGpuEndPoint)) {
        PRINT_ERROR("%d %d %d %d", "Detected LWLink connection SwitchID: %d Port: %d to GPU Index: %d Port: %d not found in topology file",
                    switchEndPointInfo.physicalId, switchEndPointInfo.linkIndex, gpuEndPointInfo.enumIndex, gpuEndPointInfo.linkIndex);
        return false;
    }

    // we have a connection from this switch:port in the topology
    return true;
}

bool
DcgmFMTopologyValidator::getLWSwitchInfoByLWLinkDevInfo(uint32 nodeId, 
                                                        DcgmFMLWLinkDevInfo &lwLinkDevInfo,
                                                        DcgmFMLWSwitchInfo &lwSwitchInfo)
{
    DcgmFMLWSwitchInfoMap::iterator it = mGfm->mLwswitchInfoMap.find(nodeId);
    if (it == mGfm->mLwswitchInfoMap.end()) {
        return false;
    }

    DcgmFMLWSwitchInfoList switchList = it->second;
    DcgmFMLWSwitchInfoList::iterator jit;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        DcgmFMLWSwitchInfo tempInfo = (*jit);
        lwlink_pci_dev_info lwlinkPciInfo = lwLinkDevInfo.getDevicePCIInfo();
        if ((tempInfo.pciInfo.domain == lwlinkPciInfo.domain) &&
             (tempInfo.pciInfo.bus == lwlinkPciInfo.bus) &&
             (tempInfo.pciInfo.device == lwlinkPciInfo.device) &&
             (tempInfo.pciInfo.function == lwlinkPciInfo.function)) {
            lwSwitchInfo = tempInfo;
            return true;
        }
    }

    return false;
}

bool
DcgmFMTopologyValidator::getGpuInfoByLWLinkDevInfo(uint32 nodeId,
                                                   DcgmFMLWLinkDevInfo &lwLinkDevInfo,
                                                   DcgmFMGpuInfo &gpuInfo)
{
    DcgmFMGpuInfoMap::iterator it = mGfm->mGpuInfoMap.find(nodeId);
    if (it == mGfm->mGpuInfoMap.end()) {
        return false;
    }

    DcgmFMGpuInfoList gpuList = it->second;
    DcgmFMGpuInfoList::iterator jit;
    for (jit = gpuList.begin(); jit != gpuList.end(); jit++) {
        DcgmFMGpuInfo tempInfo = (*jit);
        lwlink_pci_dev_info lwlinkPciInfo = lwLinkDevInfo.getDevicePCIInfo();
        if ((tempInfo.pciInfo.domain == lwlinkPciInfo.domain) &&
             (tempInfo.pciInfo.bus == lwlinkPciInfo.bus) &&
             (tempInfo.pciInfo.device == lwlinkPciInfo.device) &&
             (tempInfo.pciInfo.function == lwlinkPciInfo.function)) {
            gpuInfo = tempInfo;
            return true;
        }
    }

    return false;
}

bool
DcgmFMTopologyValidator::getAccessConnGpuLinkMaskInfo(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                                      uint64 &gpuEnabledLinkMask)
{
    DcgmLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    DcgmLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    DcgmFMLWLinkDevInfo end0LWLinkDevInfo; 
    DcgmFMLWLinkDevInfo end1LWLinkDevInfo;

    // get device details as seen by LWLinkCoreLib driver.
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint0.nodeId, endPoint0.gpuOrSwitchId, end0LWLinkDevInfo);
    mGfm->mLWLinkDevRepo.getDeviceInfo(endPoint1.nodeId, endPoint1.gpuOrSwitchId, end1LWLinkDevInfo);

    // verify that the connection is access type.
    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch &&
        end1LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        // trunk connection
        return false;
    }

    // access connection. get the corresponding GPU's enabled link mask
    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_gpu) {
        // end0 is GPU side
        gpuEnabledLinkMask = end0LWLinkDevInfo.getEnabledLinkMask();
    } else {
        // end1 is GPU side
        gpuEnabledLinkMask = end1LWLinkDevInfo.getEnabledLinkMask();
    }

    return true;
}

/*
 * Get the number of connections on the GPU
 */
int
DcgmFMTopologyValidator::getGpuConnNum(uint32 nodeId, uint32_t physicalIdx)
{
    int num = 0;
    DcgmFMGpuLWLinkConnMatrixMap::iterator it = mGfm->mGpuLWLinkConnMatrix.find(nodeId);

    if (it == mGfm->mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return num;
    }

    DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
    DcgmFMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        DcgmFMGpuLWLinkConnMatrix gpuConnInfo;
        gpuConnInfo = *jit;
        if (gpuConnInfo.gpuPhyIndex == physicalIdx) {
            for ( int i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_GPU; i++ ) {
                if (gpuConnInfo.linkConnStatus[i] == true) {
                    num++;
                }
            }
            return num;
        }
    }

    return num;
}

/*
 * find the max number of connections among all GPU
 */
int
DcgmFMTopologyValidator::getMaxGpuConnNum(void)
{
    std::map<GpuKeyType,lwswitch::gpuInfo*>::iterator it = mGfm->mpParser->gpuCfg.begin();
    int num = 0, maxNum = 0;

    for (; it != mGfm->mpParser->gpuCfg.end(); it++ ) {
        GpuKeyType key = it->first;

        num = getGpuConnNum(key.nodeId, key.physicalId);
        maxNum = ( num > maxNum ) ? num : maxNum;
    }

    return maxNum;
}

/*
 * Disable GPUs that are in the parser, but
 *  - not detected via lwlink connections
 *  - detected, but not all lwlinks up
 */
int
DcgmFMTopologyValidator::disableGpus(DcgmFMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
{
    GpuKeyType key;
    int disabledGpuCount = 0;
    int maxConn = getMaxGpuConnNum();


    std::map<GpuKeyType,lwswitch::gpuInfo*>::iterator it = mGfm->mpParser->gpuCfg.begin();

    for (; it != mGfm->mpParser->gpuCfg.end(); it++ ) {
        GpuKeyType key = it->first;

        if ( getGpuConnNum(key.nodeId, key.physicalId) < maxConn )
        {
            // GPU is detected, but number of connections is fewer
            mGfm->mpParser->disableGpu(key);
            PRINT_INFO("%d/%d(%d/%d)", "Disable GPU %d/%d(%d/%d).", key.nodeId, key.physicalId, getGpuConnNum(key.nodeId, key.physicalId), maxConn);
            disabledGpuCount++;
            uint32_t enumIndx;
            if (mGfm->getGpuEnumIndex(key.nodeId, key.physicalId, enumIndx))
            {
                // the GPU is visible with valid physicalId to enumIndex mapping
                // it is disabled due to the reason that it has less number of access links than other GPUs
                FM_SYSLOG_NOTICE("Disabling GPU physical id:%d index:%d from routing as all the LWLinks are not trained to High Speed",
                                 key.physicalId, enumIndx);
            }
        }
    }

    return disabledGpuCount;
}

/*
 * Disable Switches that are in the parser, but
 *  - not detected via lwlink connections
 */
int
DcgmFMTopologyValidator::disableSwitches(void)
{
    SwitchKeyType key;
    int disabledSwitchCount = 0;
    std::map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;

    if ( mGfm->mpParser->isSwtichGpioPresent() == false )
    {
        // gpio physicalId is not in the topology file
        // cannot compare discovered switches against the ones in the topology file
        return disabledSwitchCount;
    }

    for (it = mGfm->mpParser->lwswitchCfg.begin();
         it != mGfm->mpParser->lwswitchCfg.end();
         it++ ) {

        key = it->first;
        bool found = false;

        if (mGfm->mLwswitchInfoMap.find(key.nodeId) != mGfm->mLwswitchInfoMap.end())
        {
            DcgmFMLWSwitchInfoList switchList = mGfm->mLwswitchInfoMap[key.nodeId];
            DcgmFMLWSwitchInfoList::iterator jit;

            for (jit = switchList.begin(); jit != switchList.end(); jit++)
            {
                DcgmFMLWSwitchInfo switchInfo = (*jit);
                if ( key.physicalId == switchInfo.physicalId)
                {
                    found = true;
                    break;
                }
            }
        }

        // configured switch is not in DcgmFMLWSwitchInfoList, disable it
        if (!found) {
            PRINT_INFO("%d/%d", "Disable switch %d/%d.", key.nodeId, key.physicalId);
            mGfm->mpParser->disableSwitch(key);
            disabledSwitchCount++;
        }
    }

    return disabledSwitchCount;
}

bool
DcgmFMTopologyValidator::isIntraNodeTrunkConnsActive(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo)
{
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return as success
        return true;
    }

    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // continue if the connection is not lwlink trunk connection
        if (!isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }
        // trunk connection, validate connection state
        if (!lwLinkConn->isConnTrainedToActive()) {
            std::stringstream outStr;
            lwLinkConn->dumpConnAndStateInfo( &outStr, mGfm->mLWLinkDevRepo);
            PRINT_ERROR("%s", "LWLink intranode trunk connection is not trained to ACTIVE: %s\n",
                        outStr.str().c_str());
            return false;
        }
    }

    // all the detected intranode trunk connections are Active for the specified node
    return true;    
}

bool
DcgmFMTopologyValidator::isAccessConnsActive(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo)
{
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return as success
        return true;
    }

    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // skip if this is an lwlink trunk connection
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }
        // access connection, validate connection state
        if (!lwLinkConn->isConnTrainedToActive()) {
            std::stringstream outStr;
            lwLinkConn->dumpConnAndStateInfo( &outStr, mGfm->mLWLinkDevRepo);
            PRINT_ERROR("%s", "LWLink access connection is not trained to ACTIVE: %s\n",
                        outStr.str().c_str());
            return false;
        }
    }

    // all the detected access connections are Active for the specified node
    return true;
}

bool
DcgmFMTopologyValidator::isInterNodeTrunkConnsActive(void)
{
    LWLinkInterNodeConns::iterator jit;
    LWLinkInterNodeConns interConnList = mGfm->mLWLinkConnRepo.getInterConnections();
    for ( jit = interConnList.begin(); jit != interConnList.end(); jit++ ) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;

        // all the inter node connections should be trunk connections as of now,
        // so skipping explict isLWLinkTrunkConnection() check

        // trunk connection, validate connection state
        if (!lwLinkConn->isConnTrainedToActive()) {
            std::stringstream outStr;
            lwLinkConn->dumpConnAndStateInfo( &outStr, mGfm->mLWLinkDevRepo);
            PRINT_ERROR("%s", "LWLink internode trunk connection is not trained to ACTIVE: %s\n",
                        outStr.str().c_str());
            return false;
        }
    }

    // all the detected internode trunk connection are Active
    return true;
}

void
DcgmFMTopologyValidator::checkLWSwitchLinkInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    // iterate for all the nodes
    for ( it = mGfm->mpParser->NodeCfg.begin(); it != mGfm->mpParser->NodeCfg.end(); it++ ) {
        NodeConfig *pNode = it->second;
        DcgmFMLWLinkDevInfoList devList;
        mGfm->mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        DcgmFMLWLinkDevInfoList::iterator jit;
        // iterate for all the device in this node
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            DcgmFMLWLinkDevInfo devInfo = (*jit);
            if (devInfo.getDeviceType () == lwlink_device_type_lwswitch) {
                // found an LWSwitch device, get its link initialization status
                checkLWSwitchDeviceLinkInitStatus(  pNode->nodeId, devInfo );
            }
        }
    }
}

void
DcgmFMTopologyValidator::checkLWSwitchDeviceLinkInitStatus(uint32_t nodeId,
                                                           DcgmFMLWLinkDevInfo &devInfo)
{
    if ( !devInfo.isAllLinksInitialized() ) {
        // found initialization failed links, check whether it is physically connected as some links
        // may not be wired ( DGX-2 bare metal have 2 unused links)
        std::list<uint32> initFailedLinks;
        std::list<uint32>::iterator it;
        devInfo.getInitFailedLinksIndex( initFailedLinks );
        for ( it = initFailedLinks.begin(); it != initFailedLinks.end(); it++ ) {
            uint32 linkIndex = (*it);
            DcgmFMLWSwitchInfo switchInfo;
            getLWSwitchInfoByLWLinkDevInfo(nodeId, devInfo, switchInfo);
            if (isSwitchPortConnected(nodeId, switchInfo.physicalId, linkIndex)) {
                // log initialization failures to syslog
                std::stringstream outStr;
                outStr << "LWLink initialization failed for LWSwitch physical ID: " << switchInfo.physicalId << 
                          " port/link index: " << linkIndex;
                FM_SYSLOG_ERR("%s", outStr.str().c_str());
            }
        }
    }
}

void
DcgmFMTopologyValidator::checkGpuLinkInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    // iterate for all the nodes
    for ( it = mGfm->mpParser->NodeCfg.begin(); it != mGfm->mpParser->NodeCfg.end(); it++ ) {
        NodeConfig *pNode = it->second;
        DcgmFMLWLinkDevInfoList devList;
        mGfm->mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        DcgmFMLWLinkDevInfoList::iterator jit;
        // iterate for all the device in this node
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            DcgmFMLWLinkDevInfo devInfo = (*jit);
            if (devInfo.getDeviceType () == lwlink_device_type_gpu) {
                // found a GPU device, get its link initialization status
                checkGpuDeviceLinkInitStatus(  pNode->nodeId, devInfo );
            }
        }
    }
}

void
DcgmFMTopologyValidator::checkGpuDeviceLinkInitStatus(uint32_t nodeId,
                                                      DcgmFMLWLinkDevInfo &devInfo)
{
    if ( !devInfo.isAllLinksInitialized() ) {
        // for GPUs, all the enabled links should initialize success
        std::list<uint32> initFailedLinks;
        std::list<uint32>::iterator it;
        devInfo.getInitFailedLinksIndex( initFailedLinks );
        for ( it = initFailedLinks.begin(); it != initFailedLinks.end(); it++ ) {
            uint32 linkIndex = (*it);
            DcgmFMGpuInfo gpuInfo;
            getGpuInfoByLWLinkDevInfo(nodeId, devInfo, gpuInfo);
            // log initialization failures to syslog
            std::stringstream outStr;
            outStr << "LWLink initialization failed for GPU index: " << gpuInfo.gpuIndex <<
                      " port/link index " << linkIndex;
            FM_SYSLOG_ERR("%s", outStr.str().c_str());
        }
    }
}

uint32_t
DcgmFMTopologyValidator::getIntraNodeTrunkConnCount(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo)
{
    uint32_t connCount = 0;
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return count as zero
        return connCount;
    }

    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // count if the connection is trunk connection
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            connCount++;
        }
    }

    return connCount;
}

uint32_t
DcgmFMTopologyValidator::getAccessConnCount(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo)
{
    uint32_t connCount = 0;
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return count as zero
        return connCount;
    }

    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // count if the connection is not a trunk connection
        if (!isLWLinkTrunkConnection(lwLinkConn)) {
            connCount++;
        }
    }

    return connCount;

}

uint32_t
DcgmFMTopologyValidator::getInterNodeTrunkConnCount(uint32_t nodeId)
{
    uint32_t connCount = 0;
    LWLinkInterNodeConns::iterator jit;
    LWLinkInterNodeConns interConnList = mGfm->mLWLinkConnRepo.getInterConnections();

    for ( jit = interConnList.begin(); jit != interConnList.end(); jit++ ) {
        // get the specific connection
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *jit;

        // all the inter node connections should be trunk connections as of now,
        // so skipping explict isLWLinkTrunkConnection() check

        DcgmLWLinkEndPointInfo masterEndPoint = lwLinkConn->getMasterEndPointInfo();
        DcgmLWLinkEndPointInfo slaveEndPoint = lwLinkConn->getSlaveEndPointInfo();

        // count it if either master end or slave end of the connection belongs to
        // the specified node
        if ((masterEndPoint.nodeId == nodeId) || (slaveEndPoint.nodeId == nodeId)) {
            connCount++;
        }
    }

    return connCount;
}

bool
DcgmFMTopologyValidator::isSwitchPortConnected(uint32_t nodeId, uint32_t physicalId, uint32_t linkIndex)
{
    if (!mGfm->mpParser->isSwtichGpioPresent()) {
        // we can't validate if there is no switch GPIO based ID, treat it as no connection
        return false;
    }

    TopologyLWLinkConnEndPoint switchEndPoint;
    TopologyLWLinkConn topoLWLinkConn;
    switchEndPoint.nodeId = nodeId;
    switchEndPoint.lwswitchOrGpuId = physicalId;
    switchEndPoint.portIndex = linkIndex;

    if (!getTopologyLWLinkConnBySwitchEndPoint(switchEndPoint, topoLWLinkConn)) {
        // no connection from the specified switch:port combinaion
        return false;
    }

    // if a trunk link is enabled in LWSwitch, it is expected to have a connection
    // in all the VM cases (only for 16GPU VM as of now)
    if (topoLWLinkConn.connType != ACCESS_PORT_GPU) {
        return true;
    }

    // We have an access connection from the switch:port combinaion.
    // Due to LWSwitch even-odd pair combination, some LWSwitch links may be enabled in certain VM combination,
    // without corresponding GPU side link enabled. So cross-check with GPU link mask to disable unwanted
    // error/warning messages.
    TopologyLWLinkConnEndPoint gpuEndPoint;
    // find the corresponding GPU endpoint side
    if (topoLWLinkConn.localEnd== switchEndPoint) 
        gpuEndPoint = topoLWLinkConn.farEnd;
    else
        gpuEndPoint = topoLWLinkConn.localEnd;

    DcgmFMGpuLWLinkConnMatrixMap::iterator it = mGfm->mGpuLWLinkConnMatrix.find(gpuEndPoint.nodeId);
    if (it == mGfm->mGpuLWLinkConnMatrix.end()) {
        // no GPU entry for the specified node
        return false;
    }

    DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
    DcgmFMGpuLWLinkConnMatrixList::iterator jit;
    for (jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++) {
        DcgmFMGpuLWLinkConnMatrix gpuInfo = *jit;
        if (gpuInfo.gpuPhyIndex == gpuEndPoint.lwswitchOrGpuId) {
            // found the matching GPU, check whether corresponding link is enabled
            if (!(gpuInfo.enabledLinkMask & ((uint64)1 << gpuEndPoint.portIndex))) {
                // specified GPU link is not enabled, so don't count this connection
                return false;
            } else {
                // specified GPU link is enabled, treat this as a valid connection.
                return true;
            }
        }
    }

    // default cases like the GPU is not visible in the VM partition, ie DcgmFMGpuLWLinkConnMatrixMap
    // don't have the matching topology connection GPU.
    return false;
}

