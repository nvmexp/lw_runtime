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
#include <sstream>
#include <vector>

#include "fm_log.h"
#include "GlobalFabricManager.h"
#include "FMTopologyValidator.h"
#include "FMCommonTypes.h"
#include "FMDeviceProperty.h"
#include <g_lwconfig.h>


FMTopologyValidator::FMTopologyValidator(GlobalFabricManager *pGfm)
{
    mGfm = pGfm;
};

FMTopologyValidator::~FMTopologyValidator()
{
    // nothing as of now
};

int
FMTopologyValidator::validateTopologyInfo(void)
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

    // gpu count validation is not strictly enforced as some gpus may be excluded
    // as part of degraded mode and RM may not always return correct number of
    // excluded gpu count (such gpu pci enumeration itself fails)

    for (nodeit = mGfm->mpParser->NodeCfg.begin(); nodeit!= mGfm->mpParser->NodeCfg.end(); nodeit++) {
        NodeConfig *pNode = nodeit->second;
        uint32_t nodeId = pNode->nodeId;

        FMGpuInfoList gpuInfo = mGfm->mGpuInfoMap[nodeId];
        uint32_t gpuCount = gpuInfo.size();

        FMExcludedGpuInfoList excludedGpuInfo = mGfm->mExcludedGpuInfoMap[nodeId];
        uint32_t excludedGpuCount = excludedGpuInfo.size();

        FMLWSwitchInfoList switchInfo = mGfm->mLwswitchInfoMap[nodeId];
        uint32_t switchCount = switchInfo.size();


        // count all the intra node trunk connections discovered
        uint32_t lwLinkIntraTrunkConnCount = getIntraNodeTrunkConnCount(nodeId, mGfm->mLWLinkConnRepo);

        uint32_t lwLinkInterTrunkConnCount = 0;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        // count all the inter node trunk connections discovered
        lwLinkInterTrunkConnCount = getInterNodeTrunkConnCount(nodeId);
#endif
        // No partitions defined for multi-node topologies so ignore if this doesn't match
        if (
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            !mGfm->isMultiNodeMode() &&
#endif
            !validateNodeTopologyInfo(pNode, switchCount, lwLinkIntraTrunkConnCount,
                                      lwLinkInterTrunkConnCount, gpuCount, excludedGpuCount)
           ) {
            FM_LOG_ERROR("detected topology is not matching with system fabric topology information");
            FM_SYSLOG_ERR("detected topology is not matching with system fabric topology information");            
            retVal = false; // continue validating other nodes, but return status as failure.
        }
    }

    return retVal;
}

int
FMTopologyValidator::validateNodeTopologyInfo(NodeConfig *pNodeCfg, uint32_t switchCount,
                                              uint32_t lwLinkIntraTrunkConnCount, uint32_t lwLinkInterTrunkConnCount,
                                              uint32_t gpuCount, uint32_t excludedGpuCount)
{
    int idx = 0;

    // check for bare metal partition
    for (idx = 0; idx < pNodeCfg->partitionInfo.baremetalinfo_size(); idx++) {
        const bareMetalPartitionInfo &bareMetalInfo = pNodeCfg->partitionInfo.baremetalinfo(idx);
        const partitionMetaDataInfo &metaDataInfo = bareMetalInfo.metadata();
        // fprintf(stderr, "%s %d\n", "Number of switches: ", switchCount);
        // fprintf(stderr, "%s %d\n", "Number of switches from metadata info: ", metaDataInfo.switchcount());
        if ((metaDataInfo.switchcount() == switchCount) &&
            (metaDataInfo.lwlinkintratrunkconncount() == lwLinkIntraTrunkConnCount) &&
            (metaDataInfo.lwlinkintertrunkconncount() == lwLinkInterTrunkConnCount)) {
            // matched with bare metal partition
            // gpu count validation is not strictly enforced, log a notice for any mismatch
            if (metaDataInfo.gpucount() != (gpuCount + excludedGpuCount)) {
//                SYSLOG_NOTICE("detected GPU count is not matching with topology");
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
            if (metaDataInfo.gpucount() != (gpuCount + excludedGpuCount)) {
//                SYSLOG_NOTICE("detected GPU count is not matching with topology");
            }
            return true;
        }
    }

    // no match case
    return false;
}

bool
FMTopologyValidator::isAllIntraNodeTrunkConnsActive(void)
{
    // make sure that all the LWLink trunk connections are at ACTIVE (HIGH SPEED) mode.
    // trunk connection failures are treated as fatal error.

    // for cases like GPU baseboards are connected to different motherboards,
    // trunk connections will not be discovered and will be covered in 
    // total number of access/trunk connection validations.

    // only validate all node's intra node connections
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    for (nodeit = mGfm->mpParser->NodeCfg.begin(); nodeit!= mGfm->mpParser->NodeCfg.end(); nodeit++) {
        NodeConfig *pNode = nodeit->second;
        uint32_t nodeId = pNode->nodeId;

        //
        // In Volta/Willow baremetal, we don't support advanced degraded modes. So explicit topology validation is  
        // done to log non active LWLink connections, non-detected LWLink connections etc. In Limerock 
        // and above, these information is logged as part of degraded mode handling
        //
        if (mGfm->getSwitchArchType() == LWSWITCH_ARCH_TYPE_SV10 && mGfm->getFabricMode() == FM_MODE_BAREMETAL) {
            checkForNonDetectedLWLinkConns(nodeId);
        }

        if (!isIntraNodeTrunkConnsActive(nodeId, mGfm->mLWLinkConnRepo, mGfm->mLWLinkDevRepo)) {
            // error already logged
            return false;
        }
    }

    // all the detected trunk connections are Active
    return true;
}

void
FMTopologyValidator::checkForNonDetectedLWLinkConns(int nodeId)
{
    FMLWLinkDevInfoList devList;
    mGfm->mLWLinkDevRepo.getDeviceList( nodeId, devList );

    FMLWLinkDevInfoList::iterator dit;
    // iterate for all the device in this node
    for ( dit = devList.begin(); dit != devList.end(); dit++ ) {
        FMLWLinkDevInfo devInfo = (*dit);
        std::list<uint32> missingConnLinkIndex; 

        if (devInfo.getDeviceType() == lwlink_device_type_lwswitch) {
            FMLWSwitchInfo switchInfo;

            std::list<uint32> initFailedLinks;
            devInfo.getNonActiveLinksIndex( initFailedLinks, missingConnLinkIndex );

            if (missingConnLinkIndex.size() == 0) {
                // no missing connections to log
                continue;
            }

            if (!getLWSwitchInfoByLWLinkDevInfo(nodeId, devInfo, switchInfo)) {
                continue;
            }

            SwitchKeyType switchKey;
            switchKey.nodeId = nodeId;
            switchKey.physicalId = switchInfo.physicalId;

            logNonDetectedLWLinkConns(missingConnLinkIndex, switchKey.nodeId, switchKey.physicalId, lwlink_device_type_lwswitch);
        } else {
            GpuKeyType key;
            key.nodeId = nodeId;

            std::list<uint32_t> failedPorts;
            // all non active links are considered failed, unless they are in contain mode
            // links could enter contain mode even if neighboring links have failed
            // such links will be not be considered as failed links
            devInfo.getFailedNonContainLinksIndex(failedPorts, missingConnLinkIndex);
            if (missingConnLinkIndex.size() == 0) {
                // no missing connections to log
                continue;
            }

            FMGpuInfo_t gpuInfo;
            getGpuInfoByLWLinkDevInfo(nodeId, devInfo, gpuInfo);
            if (!mGfm->getGpuPhysicalId(nodeId, gpuInfo.uuid.bytes, key.physicalId)) {
                // the device uuid is not found
                continue;
            }

            // log information about missing links
            logNonDetectedLWLinkConns(missingConnLinkIndex, key.nodeId, key.physicalId, lwlink_device_type_gpu);
        }
    }
}

void
FMTopologyValidator::logNonDetectedLWLinkConns(std::list<uint32> &missingConnLinkIndex, uint32_t nodeId,
                                               uint32_t physicalId, uint64 deviceType)
{
    std::list<uint32>::iterator it;
    for (it = missingConnLinkIndex.begin(); it != missingConnLinkIndex.end(); it++) {
        TopologyLWLinkConnEndPoint endPoint = {};
        endPoint.nodeId = nodeId;
        endPoint.lwswitchOrGpuId = physicalId;
        endPoint.portIndex = *it;
        TopologyLWLinkConn topoLWLinkConn;
        FMLWLinkDevInfo localDevInfo, farDevInfo;
        string localDeviceName, farDeviceName;

        // this logs only trunk connections that are not detected
        // this logs only once for each link
        if (deviceType == lwlink_device_type_lwswitch && 
            getTopologyLWLinkConnBySwitchEndPoint(endPoint, topoLWLinkConn) &&
            topoLWLinkConn.connType == TRUNK_PORT_SWITCH) {
            TopologyLWLinkConnEndPoint localEnd = topoLWLinkConn.localEnd;
            TopologyLWLinkConnEndPoint farEnd = topoLWLinkConn.farEnd;

            mGfm->mLWLinkDevRepo.getDeviceInfo(localEnd.nodeId, localEnd.lwswitchOrGpuId, localDevInfo);
            mGfm->mLWLinkDevRepo.getDeviceInfo(farEnd.nodeId, farEnd.lwswitchOrGpuId, farDevInfo);
            localDeviceName = localDevInfo.getDeviceName();
            farDeviceName = farDevInfo.getDeviceName();

            if (!(mGfm->isNodeDegraded(localEnd.nodeId) || mGfm->isNodeDegraded(farEnd.nodeId))) {
            #if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
                FM_LOG_ERROR("LWLink trunk connection not detected NodeId:%d DeviceName:%s (PhysicalId:%d) "
                             "OsfpPort:%d LWLinkIndex:%d <=======> NodeId:%d DeviceName:%s (PhysicalId:%d) "
                             "OsfpPort:%d LWLinkIndex:%d ", localEnd.nodeId, localDeviceName.c_str(),
                             localEnd.lwswitchOrGpuId, mGfm->mpParser->getOsfpPortNumForLinkIndex(localEnd.portIndex),
                             localEnd.portIndex, farEnd.nodeId, farDeviceName.c_str(), farEnd.lwswitchOrGpuId,
                             mGfm->mpParser->getOsfpPortNumForLinkIndex(farEnd.portIndex), farEnd.portIndex);
            #else
                FM_LOG_ERROR("LWLink trunk connection not detected DeviceName:%s (PhysicalId:%d) LWLinkIndex:%d "
                             "<=======> DeviceName:%s (PhysicalId:%d) LWLinkIndex:%d",
                             localDeviceName.c_str(), localEnd.lwswitchOrGpuId, localEnd.portIndex, farDeviceName.c_str(),
                             farEnd.lwswitchOrGpuId, farEnd.portIndex);
            #endif
            }
        }

        // this logs access link connections that are missed
        // logs only once for each link, as we dont log any access
        // link related stuff from the switch side
        // doesnt log missing connections for MIG enabled GPU/Blacklisted GPUs as they
        // dont result in degrading
        else if (deviceType == lwlink_device_type_gpu && 
                 getTopologyLWLinkConnByGpuEndPoint(endPoint, topoLWLinkConn) &&
                 topoLWLinkConn.connType == ACCESS_PORT_GPU) {
            TopologyLWLinkConnEndPoint localEnd = topoLWLinkConn.localEnd;
            TopologyLWLinkConnEndPoint farEnd = topoLWLinkConn.farEnd;

            mGfm->mLWLinkDevRepo.getDeviceInfo(localEnd.nodeId, localEnd.lwswitchOrGpuId, localDevInfo);
            localDeviceName = localDevInfo.getDeviceName();

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            FM_LOG_ERROR("LWLink access connection not detected NodeId:%d LWSwitchDeviceName:%s (PhysicalId:%d), LWLinkIndex:%d <=======> "
                         "NodeId:%d GPUPhysicalId:%d, LWLinkIndex:%d", 
                         localEnd.nodeId, localDeviceName.c_str(), localEnd.lwswitchOrGpuId, localEnd.portIndex,
                         farEnd.nodeId, farEnd.lwswitchOrGpuId, farEnd.portIndex);
#else
            FM_LOG_ERROR("Access LWLink connection not detected LWSwitchDeviceName:%s (PhysicalId:%d), LWLinkIndex:%d <=======> "
                         "GPUPhysicalId:%d, LWLinkIndex:%d", 
                         localDeviceName.c_str(), localEnd.lwswitchOrGpuId, localEnd.portIndex,
                         farEnd.lwswitchOrGpuId, farEnd.portIndex);
#endif
        }
    }
}

void
FMTopologyValidator::checkLWLinkInitStatusForAllDevices()
{
    checkLWSwitchLinkInitStatus();
    checkGpuLinkInitStatus();

    //also check if access and trunk intra-node links are trained to active and log any potential failures
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    for ( it = mGfm->mpParser->NodeCfg.begin(); it != mGfm->mpParser->NodeCfg.end(); it++ ) {
        NodeConfig *pNode = it->second;
        isAccessConnsActive(pNode->nodeId, mGfm->mLWLinkConnRepo, mGfm->mLWLinkDevRepo);
        isIntraNodeTrunkConnsActive(pNode->nodeId, mGfm->mLWLinkConnRepo, mGfm->mLWLinkDevRepo);
    }

}

bool
FMTopologyValidator::mapGpuIndexByLWLinkConns(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
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
        FMGpuLWLinkConnMatrixList connMatrix;
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
FMTopologyValidator::mapGpuIndexForOneNode(uint32_t nodeId,
                                           FMGpuLWLinkConnMatrixList &connMatrix)
{
    // create GPU Physical ID to local index for the specified node.

    LWLinkIntraConnMap tempIntraConnMap = mGfm->mLWLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);
    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator jit;

    // iterate for all the connections in the node.
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        FMDetailedEndPointInfo switchEndPointInfo;
        FMDetailedEndPointInfo gpuEndPointInfo;

        // no need to create mapping for trunk connections
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }

        //
        // skip if the connection is not trained to active for Willow based systems. For such systems
        // the only degradation is to remove corresponding GPU from LWLink P2P. For that, later we will
        // compare with the max number of LWLinks trained across all the GPUs to a given GPU's trained
        // LWLink numbers. For systems with advanced degraded modes (LimeRock) degradation will happen
        // as part of degraded mode processing.
        //
        if (mGfm->getSwitchArchType() == LWSWITCH_ARCH_TYPE_SV10) {
            if (!lwLinkConn->isConnTrainedToActive()) {
                continue;
            }
        }

        // get the corresponding switch and gpu information from the LWLink Connection
        FMLWSwitchInfo switchInfo;
        FMGpuInfo_t gpuInfo;
        if (!getAccessConnDetailedEndPointInfo(lwLinkConn, switchEndPointInfo, gpuEndPointInfo,
                                               switchInfo, gpuInfo)) {
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

        // update/populate our GPU connection matrix with the physical id
        // specified in the topology and was detected by the driver.
        if (!updateGpuConnectionMatrixInfo(connMatrix, gpuEndPointInfo,
             topoGpuEndPoint, gpuInfo.uuid)) {
            // error already logged. break from further parsing          
            return false;
        }
    }

    return true;
}

bool
FMTopologyValidator::updateGpuConnectionMatrixInfo(FMGpuLWLinkConnMatrixList &connMatrixList,
                                                   FMDetailedEndPointInfo gpuEndPointInfo,
                                                   TopologyLWLinkConnEndPoint topoGpuEndPoint,
                                                   FMUuid_t &uuid)
{
    FMGpuLWLinkConnMatrixList::iterator it;
    FMGpuLWLinkConnMatrix gpuConnMatrix = {0};

    for (it = connMatrixList.begin(); it != connMatrixList.end(); it++) {
        FMGpuLWLinkConnMatrix tempMatrix = *it;
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
            FM_LOG_ERROR("duplicate LWLink connection " NODE_ID_LOG_STR ":%d GPUPhysicalId:%d enumIndex:%d LWLinkIndex:%d\n",
                         gpuEndPointInfo.nodeId, topoGpuEndPoint.lwswitchOrGpuId, gpuEndPointInfo.enumIndex, topoGpuEndPoint.portIndex);
            return false;
        }
        // mark this link as connected.
        gpuConnMatrix.linkConnStatus[topoGpuEndPoint.portIndex] = true;
    } else {
        // first connection for this GPU
        gpuConnMatrix.gpuPhyIndex = topoGpuEndPoint.lwswitchOrGpuId;
        gpuConnMatrix.gpuEnumIndex = gpuEndPointInfo.enumIndex;        
        gpuConnMatrix.linkConnStatus[topoGpuEndPoint.portIndex] = true;
        strncpy(gpuConnMatrix.uuid.bytes, uuid.bytes, FM_UUID_BUFFER_SIZE - 1);
    }

    // re-add/insert the connection info to the list
    connMatrixList.push_back(gpuConnMatrix);

    return true;
}

bool
FMTopologyValidator::getTopologyLWLinkConnGpuEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
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
FMTopologyValidator::getTopologyLWLinkConnBySwitchEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
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
FMTopologyValidator::getTopologyLWLinkConnByGpuEndPoint(TopologyLWLinkConnEndPoint &gpuEndPoint,
                                                        TopologyLWLinkConn &topoLWLinkConn)
{
    TopologyLWLinkConnMap::iterator it = mGfm->mpParser->lwLinkConnMap.find(gpuEndPoint.nodeId);

    if (it == mGfm->mpParser->lwLinkConnMap.end()) {
        // no connection for this node
        return false;
    }

    TopologyLWLinkConnList connList = it->second;
    TopologyLWLinkConnList::iterator jit;
    for (jit = connList.begin(); jit != connList.end(); jit++) {
        TopologyLWLinkConn conn = (*jit);
        // lwLinkConnMap is built as switch at localEnd, gpu at farEnd
        if ((conn.farEnd.nodeId == gpuEndPoint.nodeId) &&
            (conn.farEnd.lwswitchOrGpuId == gpuEndPoint.lwswitchOrGpuId) &&
            (conn.farEnd.portIndex == gpuEndPoint.portIndex)) {
            topoLWLinkConn = conn;
            return true;
        }
    }

    // not found any matching topology connection
    return false;
}

bool
FMTopologyValidator::isLWLinkTrunkConnection(FMLWLinkDetailedConnInfo *lwLinkConn)
{
    FMLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    FMLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    FMLWLinkDevInfo end0LWLinkDevInfo; 
    FMLWLinkDevInfo end1LWLinkDevInfo;

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
FMTopologyValidator::getAccessConnDetailedEndPointInfo(FMLWLinkDetailedConnInfo *lwLinkConn,
                                                       FMDetailedEndPointInfo &switchEndPointInfo,
                                                       FMDetailedEndPointInfo &gpuEndPointInfo,
                                                       FMLWSwitchInfo &switchInfo,
                                                       FMGpuInfo_t &gpuInfo)
{
    // translate and LWLink connection endpoints seen by LWLinkCoreLib driver
    // to the corresponding devices reported by LWSwtich Driver and RM driver.

    FMLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    FMLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    FMLWLinkDevInfo end0LWLinkDevInfo; 
    FMLWLinkDevInfo end1LWLinkDevInfo;

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
    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        // assign switch endpoint details accordingly
        if (!getLWSwitchInfoByLWLinkDevInfo(endPoint0.nodeId, end0LWLinkDevInfo, switchInfo)) {
            lwlink_pci_dev_info pciInfo = end0LWLinkDevInfo.getDevicePCIInfo();
            FM_LOG_ERROR("no matching LWSwitch device for " NODE_ID_LOG_STR ":%u driver PCI BDF %x:%x:%x:%x",
                         endPoint0.nodeId, pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
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
            FM_LOG_ERROR("no matching GPU device for " NODE_ID_LOG_STR ":%u driver PCI BDF %x:%x:%x:%x",
                         endPoint1.nodeId, pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        gpuEndPointInfo.enumIndex = gpuInfo.gpuIndex;
        gpuEndPointInfo.nodeId = endPoint1.nodeId;
        gpuEndPointInfo.linkIndex = endPoint1.linkIndex;
    } else {
        if (!getLWSwitchInfoByLWLinkDevInfo(endPoint1.nodeId, end1LWLinkDevInfo, switchInfo)) {
            lwlink_pci_dev_info pciInfo = end1LWLinkDevInfo.getDevicePCIInfo();
            FM_LOG_ERROR("no matching LWSwitch device for " NODE_ID_LOG_STR ":%u driver PCI BDF %x:%x:%x:%x",
                         endPoint1.nodeId, pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
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
            FM_LOG_ERROR("no matching GPU device for " NODE_ID_LOG_STR ":%u driver PCI BDF %x:%x:%x:%x",
                         endPoint0.nodeId, pciInfo.domain, pciInfo.bus, pciInfo.device, pciInfo.function);
            return false;
        }
        gpuEndPointInfo.enumIndex = gpuInfo.gpuIndex;
        gpuEndPointInfo.nodeId = endPoint0.nodeId;
        gpuEndPointInfo.linkIndex = endPoint0.linkIndex;
    }

    return true;
}

bool
FMTopologyValidator::verifyAccessConnection(FMLWLinkDetailedConnInfo *lwLinkConn)
{
    // verify whether this access connection is specified in our topology
    FMDetailedEndPointInfo switchEndPointInfo;
    FMDetailedEndPointInfo gpuEndPointInfo;
    FMLWSwitchInfo switchInfo;
    FMGpuInfo_t gpuInfo;

    getAccessConnDetailedEndPointInfo(lwLinkConn, switchEndPointInfo, gpuEndPointInfo,
                                      switchInfo, gpuInfo);

    TopologyLWLinkConnEndPoint topoSwitchEndPoint;
    TopologyLWLinkConnEndPoint topoGpuEndPoint;        
    topoSwitchEndPoint.nodeId = switchEndPointInfo.nodeId;
    topoSwitchEndPoint.lwswitchOrGpuId = switchEndPointInfo.physicalId;
    topoSwitchEndPoint.portIndex = switchEndPointInfo.linkIndex;
    // to verify the connection, check whether we have a remote connection with
    // the specified switch:port exists in the topology.
    if (!getTopologyLWLinkConnGpuEndPoint(topoSwitchEndPoint, topoGpuEndPoint)) {
        FMLWLinkDevInfo switchDevInfo;
        mGfm->mLWLinkDevRepo.getDeviceInfo(topoSwitchEndPoint.nodeId, topoSwitchEndPoint.lwswitchOrGpuId, switchDevInfo);
        string switchDeviceName = switchDevInfo.getDeviceName();

        FM_LOG_ERROR("detected LWLink connection not found in topology file " NODE_ID_LOG_STR ":%d LWSwitchDeviceName:%s (physical id:%d) LWLinkIndex:%d <=======> " NODE_ID_LOG_STR ":%d GPUIndex:%d LWLinkIndex:%d",
                     topoSwitchEndPoint.nodeId, switchDeviceName.c_str(), switchEndPointInfo.physicalId, switchEndPointInfo.linkIndex, topoGpuEndPoint.nodeId, gpuEndPointInfo.enumIndex, gpuEndPointInfo.linkIndex);
        return false;
    }

    // we have a connection from this switch:port in the topology
    return true;
}

bool
FMTopologyValidator::getLWSwitchInfoByLWLinkDevInfo(uint32 nodeId, 
                                                    FMLWLinkDevInfo &lwLinkDevInfo,
                                                    FMLWSwitchInfo &lwSwitchInfo)
{
    FMLWSwitchInfoMap::iterator it = mGfm->mLwswitchInfoMap.find(nodeId);
    if (it == mGfm->mLwswitchInfoMap.end()) {
        return false;
    }

    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        FMLWSwitchInfo tempInfo = (*jit);
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
FMTopologyValidator::getGpuInfoByLWLinkDevInfo(uint32 nodeId,
                                                   FMLWLinkDevInfo &lwLinkDevInfo,
                                                   FMGpuInfo_t &gpuInfo)
{
    FMGpuInfoMap::iterator it = mGfm->mGpuInfoMap.find(nodeId);
    if (it == mGfm->mGpuInfoMap.end()) {
        return false;
    }

    FMGpuInfoList gpuList = it->second;
    FMGpuInfoList::iterator jit;
    for (jit = gpuList.begin(); jit != gpuList.end(); jit++) {
        FMGpuInfo_t tempInfo = (*jit);
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
FMTopologyValidator::getAccessConnGpuLinkMaskInfo(FMLWLinkDetailedConnInfo *lwLinkConn,
                                                      uint64 &gpuEnabledLinkMask)
{
    FMLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    FMLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    FMLWLinkDevInfo end0LWLinkDevInfo; 
    FMLWLinkDevInfo end1LWLinkDevInfo;

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
FMTopologyValidator::getGpuConnNum(uint32 nodeId, uint32_t physicalIdx)
{
    int num = 0;
    FMGpuLWLinkConnMatrixMap::iterator it = mGfm->mGpuLWLinkConnMatrix.find(nodeId);

    if (it == mGfm->mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return num;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        FMGpuLWLinkConnMatrix gpuConnInfo;
        gpuConnInfo = *jit;
        if (gpuConnInfo.gpuPhyIndex == physicalIdx) {
            for ( uint32_t i = 0; i < FMDeviceProperty::getLWLinksPerGpu(mGfm->getSwitchArchType()); i++ ) {
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
FMTopologyValidator::getMaxGpuConnNum(void)
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
 * get a list of LWLink unmapped GPUs
 * unmapped GPUs are the ones without a enumeration index to physical id mapping
 */
void
FMTopologyValidator::getUnmappedGpusForOneNode(uint32_t nodeId, FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap,
                                               FMGpuInfoList &unmappedGpuList)
{
    unmappedGpuList.clear();

    const FMGpuInfoMap &gpuInfoMap = mGfm->getGpuInfoMap();
    FMGpuInfoMap::const_iterator gpuInfoMapIt = gpuInfoMap.find(nodeId);
    if (gpuInfoMapIt == gpuInfoMap.end()) {
        // no GPU is detected on the node
        return;
    }
    FMGpuInfoList gpuInfoList = gpuInfoMapIt->second;

    FMGpuLWLinkConnMatrixMap::iterator connMatrixMapIt = gpuConnMatrixMap.find(nodeId);
    if (connMatrixMapIt == gpuConnMatrixMap.end()) {
        // all GPUs are unmapped on this node
        unmappedGpuList = gpuInfoList;
        return;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = connMatrixMapIt->second;
    FMGpuInfoList::iterator gpuInfoIt;
    FMGpuLWLinkConnMatrixList::iterator connMatrixIt;

    for (gpuInfoIt = gpuInfoList.begin(); gpuInfoIt != gpuInfoList.end(); gpuInfoIt++) {
        FMGpuInfo_t gpuInfo = *gpuInfoIt;
        bool foundMapping = false;

        for (connMatrixIt = connMatrixList.begin();  connMatrixIt!= connMatrixList.end(); connMatrixIt++) {
            FMGpuLWLinkConnMatrix connMatrix = *connMatrixIt;

            if (strncmp(gpuInfo.uuid.bytes, connMatrix.uuid.bytes, FM_UUID_BUFFER_SIZE) == 0) {
                // find the GPU mapping
                foundMapping = true;
                break;
            }
        }

        if (foundMapping == false) {
            // add this GPU to the unmappedGpuList
            unmappedGpuList.push_back(gpuInfo);
        }
    }
}

// check all GPU ports to see if it is connected to a port on the running system
bool
FMTopologyValidator::isGpuConnectedToSwitch(GpuKeyType key)
{
    TopologyLWLinkConnEndPoint gpuEndPoint;
    gpuEndPoint.nodeId = key.nodeId;
    gpuEndPoint.lwswitchOrGpuId = key.physicalId;

    for (gpuEndPoint.portIndex = 0;
         gpuEndPoint.portIndex < FMDeviceProperty::getLWLinksPerGpu(mGfm->getSwitchArchType());
         gpuEndPoint.portIndex++) {

        TopologyLWLinkConn topoLWLinkConn;

        if (getTopologyLWLinkConnByGpuEndPoint(gpuEndPoint, topoLWLinkConn)) {
            FMLWSwitchInfo switchInfo;

            //  lwLinkConnMap is built as switch at localEnd, GPU at farEnd
            if (mGfm->getLWSwitchInfo(key.nodeId, topoLWLinkConn.localEnd.lwswitchOrGpuId, switchInfo) &&
                ((switchInfo.enabledLinkMask & (1LL << topoLWLinkConn.localEnd.portIndex)) != 0)) {
                // the LWSwitch that this GPU connected to is present
                // and the corresponding switch port is enabled

                // do not prune this unmapped GPU
                return true;
            }
        }
    }

    return false;
}

int
FMTopologyValidator::disableGpus(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
{
    if (mGfm->getSwitchArchType() == LWSWITCH_ARCH_TYPE_SV10) {
        return disableGpusForNonDegradedMode(gpuConnMatrixMap);
    } else {
        return disableGpusForDegradedMode(gpuConnMatrixMap);
    }
}

/*
 * Disable GPUs that are in the parser, but
 *  - not detected via lwlink connections
 *  - detected, but not all lwlinks up
 */
int
FMTopologyValidator::disableGpusForNonDegradedMode(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
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
            mGfm->mpParser->disableGpu(key, mGfm->getSwitchArchType());
            FM_LOG_INFO("disabling GPU %d/%d.", key.nodeId, key.physicalId);
            disabledGpuCount++;
            uint32_t enumIndx;
            if (mGfm->getGpuEnumIndex(key.nodeId, key.physicalId, enumIndx))
            {
                // the GPU is visible with valid physicalId to enumIndex mapping
                // it is disabled due to the reason that it has less number of access links than other GPUs
                FM_LOG_INFO("disabling GPU physical id:%d index:%d from routing and LWLink P2P as all the LWLinks are not trained to High Speed",
                                 key.physicalId, enumIndx);
                FM_SYSLOG_NOTICE("disabling GPU physical id:%d index:%d from routing and LWLink P2P as all the LWLinks are not trained to High Speed",
                                 key.physicalId, enumIndx);
            }
        }
    }

    return disabledGpuCount;
}

/*
 * Disable GPUs that are in the parser, but
 *  - not detected via lwlink connections
 *  - detected, but not all lwlinks up GPUs are handled by DegradedModeMgr
 */
int
FMTopologyValidator::disableGpusForDegradedMode(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap)
{
    GpuKeyType key;
    int disabledGpuCount = 0;
    uint32_t maxNumNodes = 1;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    maxNumNodes = mGfm->mpParser->getMaxNumNodes();
#endif
    std::unique_ptr< std::vector<bool>> notMappedGpusNodeList(new std::vector<bool>(maxNumNodes));

    for (uint32_t nodeId = 0; nodeId < maxNumNodes; nodeId++) {
        FMGpuInfoList unmappedGpuList;
        getUnmappedGpusForOneNode(nodeId, gpuConnMatrixMap, unmappedGpuList);

        if (unmappedGpuList.size() == 0) {
            // all detected GPUs on this node are mapped
            // such as in bare metal or a full pass through VM
            (*notMappedGpusNodeList)[nodeId] = true;
        } else {
            // some detected GPUs on this node are unmapped
            // such as in the case of MIG enabled GPU
            (*notMappedGpusNodeList)[nodeId] = false;
        }
    }

    std::map<GpuKeyType,lwswitch::gpuInfo*>::iterator it = mGfm->mpParser->gpuCfg.begin();

    for (; it != mGfm->mpParser->gpuCfg.end(); it++ ) {
        GpuKeyType key = it->first;
        uint32_t enumIndx;

        if (!mGfm->getGpuEnumIndex(key.nodeId, key.physicalId, enumIndx))
        {
            // found a GPU in the config, but no LWLink mapping
            // prune the GPU by default

            // if the connected switch is detected, do not prune the GPU
            // the GPU could have no LWLink enabled, such as when MIG is enabled
            bool pruneGpu;
            if (isGpuConnectedToSwitch(key) == true) {
                pruneGpu = false;
            } else {
                pruneGpu = true;
            }

            if ((*notMappedGpusNodeList)[key.nodeId] || pruneGpu) {
                mGfm->mpParser->disableGpu(key, mGfm->getSwitchArchType());
                FM_LOG_INFO("disabling GPU %d/%d.", key.nodeId, key.physicalId);
                disabledGpuCount++;
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
FMTopologyValidator::disableSwitches(void)
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
            FMLWSwitchInfoList switchList = mGfm->mLwswitchInfoMap[key.nodeId];
            FMLWSwitchInfoList::iterator jit;

            for (jit = switchList.begin(); jit != switchList.end(); jit++)
            {
                FMLWSwitchInfo switchInfo = (*jit);
                if ( key.physicalId == switchInfo.physicalId)
                {
                    found = true;
                    break;
                }
            }
        }

        // configured switch is not in FMLWSwitchInfoList, disable it
        if (!found) {
            FM_LOG_INFO("Disable switch %d/%d.", key.nodeId, key.physicalId);
            mGfm->mpParser->disableSwitch(key);
            disabledSwitchCount++;
        }
    }

    return disabledSwitchCount;
}

bool
FMTopologyValidator::isIntraNodeTrunkConnsActive(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                                                 GlobalFMLWLinkDevRepo &lwLinkDevRepo)
{
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);
    bool ret = true;

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return as success
        return ret;
    }

    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // continue if the connection is not lwlink trunk connection
        if (!isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }
        // trunk connection, validate connection state
        // continue logging all the failed connections and keep the over-all status as false
        if (!lwLinkConn->isConnTrainedToActive()) {
            std::stringstream outStr;
            lwLinkConn->dumpConnAndStateInfo( &outStr, mGfm, lwLinkDevRepo);
            FM_LOG_ERROR("LWLink intranode trunk connection not trained to ACTIVE: %s\n",
                        outStr.str().c_str());
            ret = false;
        }
    }

    return ret;    
}

bool
FMTopologyValidator::isAccessConnsActive(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                                         GlobalFMLWLinkDevRepo &lwLinkDevRepo)
{
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);
    bool ret = true;

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return as success
        return ret;
    }

    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // skip if this is an lwlink trunk connection
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            continue;
        }
        // access connection, validate connection state
        // continue logging all the failed connections and keep the over-all status as false
        if (!lwLinkConn->isConnTrainedToActive()) {
            std::stringstream outStr;
            lwLinkConn->dumpConnAndStateInfo( &outStr, mGfm, lwLinkDevRepo);
            FM_LOG_ERROR("LWLink access connection is not trained to ACTIVE: %s\n",
                        outStr.str().c_str());
            ret = false;
        }
    }

    return ret;
}

bool
FMTopologyValidator::compareLinks(FMLWLinkDetailedConnInfo *discoveredConn, TopologyLWLinkConn configConn)
{
    // FMLWLinkDetailedConnInfo has FMGpuOrSwitchId_t whereas TopologyLWLinkConn has FMPhysicalId_t
    // Colwert FMGpuOrSwitchId_t to FMPhysicalId_t to compare

    // configured endpoints
    TopologyLWLinkConnEndPoint configLocalEnd = configConn.localEnd;
    TopologyLWLinkConnEndPoint configFarEnd = configConn.farEnd;

    // discovered endpoints
    FMLWLinkEndPointInfo discoveredMasterEndPoint = discoveredConn->getMasterEndPointInfo();
    FMLWLinkEndPointInfo discoveredSlaveEndPoint = discoveredConn->getSlaveEndPointInfo();
    FMLWLinkDevInfo masterEndDevInfo, slaveEndDevInfo;
    FMLWSwitchInfo masterSwitchInfo, slaveSwitchInfo;
    mGfm->mLWLinkDevRepo.getDeviceInfo( discoveredMasterEndPoint.nodeId, discoveredMasterEndPoint.gpuOrSwitchId,
                                        masterEndDevInfo );
    mGfm->mLWLinkDevRepo.getDeviceInfo( discoveredSlaveEndPoint.nodeId, discoveredSlaveEndPoint.gpuOrSwitchId, 
                                        slaveEndDevInfo );
    getLWSwitchInfoByLWLinkDevInfo( discoveredMasterEndPoint.nodeId, masterEndDevInfo, masterSwitchInfo );
    getLWSwitchInfoByLWLinkDevInfo( discoveredSlaveEndPoint.nodeId, slaveEndDevInfo, slaveSwitchInfo );

    // compare
    if( discoveredMasterEndPoint.nodeId == configLocalEnd.nodeId
        && discoveredMasterEndPoint.linkIndex == configLocalEnd.portIndex
        && masterSwitchInfo.physicalId ==  configLocalEnd.lwswitchOrGpuId 
        && discoveredSlaveEndPoint.nodeId == configFarEnd.nodeId
        && discoveredSlaveEndPoint.linkIndex == configFarEnd.portIndex
        && slaveSwitchInfo.physicalId ==  configFarEnd.lwswitchOrGpuId )
        return true;

    if( discoveredSlaveEndPoint.nodeId == configLocalEnd.nodeId
        && discoveredSlaveEndPoint.linkIndex == configLocalEnd.portIndex
        && slaveSwitchInfo.physicalId == configLocalEnd.lwswitchOrGpuId
        && discoveredMasterEndPoint.nodeId == configFarEnd.nodeId
        && discoveredMasterEndPoint.linkIndex == configFarEnd.portIndex 
        && masterSwitchInfo.physicalId == configFarEnd.lwswitchOrGpuId )
        return true;

    return false;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
FMTopologyValidator::addFailedConnection(GlobalFMLWLinkConnRepo &failedConnections, TopologyLWLinkConn configConn)
{
    FMLWLinkConnInfo conn;
    conn.masterEnd.nodeId = configConn.localEnd.nodeId;
    conn.masterEnd.linkIndex = configConn.localEnd.portIndex;
    mGfm->getLWSwitchLWLinkDriverId( configConn.localEnd.nodeId, configConn.localEnd.lwswitchOrGpuId, conn.masterEnd.gpuOrSwitchId );

    conn.slaveEnd.nodeId = configConn.farEnd.nodeId;
    conn.slaveEnd.linkIndex = configConn.farEnd.portIndex;
    mGfm->getLWSwitchLWLinkDriverId( configConn.farEnd.nodeId, configConn.farEnd.lwswitchOrGpuId, conn.slaveEnd.gpuOrSwitchId );

    failedConnections.addInterConnections( conn );
}

int
FMTopologyValidator::numActiveInternodeTrunks()
{
    int activeTrunkCount = 0;
    LWLinkInterNodeConns discoveredConnList = mGfm->mLWLinkConnRepo.getInterConnections();
    // iterate over list of discovered internode trunk connections
    for( LWLinkInterNodeConns::iterator discoveredConnIt = discoveredConnList.begin(); 
         discoveredConnIt != discoveredConnList.end(); discoveredConnIt++ )
    {
        FMLWLinkDetailedConnInfo *discoveredConn = *discoveredConnIt;
        if( isLWLinkTrunkConnection(discoveredConn) == false )
            continue; 

        if( !discoveredConn->isConnTrainedToActive() )
            continue;

        activeTrunkCount++;
    }
    return activeTrunkCount;
}

int
FMTopologyValidator::numActiveIntranodeTrunks()
{

    int activeTrunkCount = 0;
    // check in discovered intranode connections
    LWLinkIntraConnMap discoveredIntraMap = mGfm->mLWLinkConnRepo.getIntraConnections();
    // iterate over all nodes
    for( LWLinkIntraConnMap::iterator nodeIt = discoveredIntraMap.begin();
         nodeIt != discoveredIntraMap.end(); nodeIt++ )
    {
        // iterate over discovered connections within a node
        FMLWLinkDetailedConnInfoList *discoveredConn = &nodeIt->second;
        for( FMLWLinkDetailedConnInfoList::iterator discoveredConnIt = discoveredConn->begin();
             discoveredConnIt != discoveredConn->end(); discoveredConnIt++ )
        {
            FMLWLinkDetailedConnInfo *discoveredConn = *discoveredConnIt;
            if( isLWLinkTrunkConnection(discoveredConn) == false )
                continue;

            if( !discoveredConn->isConnTrainedToActive() )
                continue;
            activeTrunkCount++;
        }
    }
    return activeTrunkCount;
}

bool
FMTopologyValidator::isNodeConfigured(FMNodeId_t nodeId)
{
    NodeKeyType nodeKey;
    nodeKey.nodeId =  nodeId;
    if( mGfm->mpParser->NodeCfg.find(nodeKey) != mGfm->mpParser->NodeCfg.end() )
        return true;
    else
        return false;
}

bool
FMTopologyValidator::checkConnInDiscoveredConnections( TopologyLWLinkConn &configConn )
{
    // check in discovered internode connections
    LWLinkInterNodeConns discoveredConnList = mGfm->mLWLinkConnRepo.getInterConnections();
    // iterate over list of discovered internode trunk connections
    for( LWLinkInterNodeConns::iterator discoveredConnIt = discoveredConnList.begin(); 
         discoveredConnIt != discoveredConnList.end(); discoveredConnIt++ )
    {
        FMLWLinkDetailedConnInfo *discoveredConn = *discoveredConnIt;
        if( isLWLinkTrunkConnection(discoveredConn) == false )
            continue; 

        if( !discoveredConn->isConnTrainedToActive() )
            continue;
        
        // check if configured connection is discovered at run time 
        if( compareLinks( discoveredConn,  configConn ) == true )
        {
            std::stringstream outStr;
            discoveredConn->dumpConnAndStateInfo( &outStr, mGfm, mGfm->mLWLinkDevRepo);
            FM_LOG_DEBUG( "LWLink internode trunk connection found: %s", outStr.str().c_str() );
            return true;
        }
    }

    // check in discovered intranode connections
    LWLinkIntraConnMap discoveredIntraMap = mGfm->mLWLinkConnRepo.getIntraConnections();
    // iterate over all nodes
    for( LWLinkIntraConnMap::iterator nodeIt = discoveredIntraMap.begin();
         nodeIt != discoveredIntraMap.end(); nodeIt++ )
    {
        // iterate over discovered connections within a node
        FMLWLinkDetailedConnInfoList *discoveredConn = &nodeIt->second;
        for( FMLWLinkDetailedConnInfoList::iterator discoveredConnIt = discoveredConn->begin();
             discoveredConnIt != discoveredConn->end(); discoveredConnIt++ )
        {
            FMLWLinkDetailedConnInfo *discoveredConn = *discoveredConnIt;
            if( isLWLinkTrunkConnection(discoveredConn) == false )
                continue;

            if( !discoveredConn->isConnTrainedToActive() )
                continue;

            // check if configured connection is discovered at run time
            if( compareLinks( discoveredConn,  configConn ) == true )
            {
                std::stringstream outStr;
                discoveredConn->dumpConnAndStateInfo( &outStr, mGfm, mGfm->mLWLinkDevRepo);
                FM_LOG_DEBUG( "LWLink internode trunk connection found: %s", outStr.str().c_str() );
                return true;
            }
        }
    }

    return false;
}

bool
FMTopologyValidator::isTrunkConnsActive( GlobalFMLWLinkConnRepo &failedConnections, int &numConfigTrunks)
{
    bool retVal = true;

    numConfigTrunks = 0;

    TopologyLWLinkConnMap:: iterator it;
    // iterate over list of configured internode trunk connections for all nodes
    for( it = mGfm->mpParser->lwLinkConnMap.begin(); it != mGfm->mpParser->lwLinkConnMap.end(); it++ )
    {
        FM_LOG_DEBUG( "Internode connection list for node %d", it->first );
        TopologyLWLinkConnList configConnList = it->second;

        TopologyLWLinkConnList::iterator jit;
        // iterate over list of configured internode trunk connections for one node
        for( jit = configConnList.begin(); jit != configConnList.end(); jit++ ) {
            TopologyLWLinkConn configConn = ( *jit );
            FMLWLinkDevInfo localDevInfo, farDevInfo;
            string localDeviceName, farDeviceName;

            // Since connections are repeated twice (once for each end as master)
            // Using only smaller node ID as master prevent redundant checks
            // Also checks for loopout/loopout trunk connections
            if( ( configConn.connType != TRUNK_PORT_SWITCH) || ( configConn.localEnd.nodeId > configConn.farEnd.nodeId) )
                continue;
            
            // If node on either end is degraded via fabric_node_config, the connection is not checked
            if( !isNodeConfigured( configConn.localEnd.nodeId ) || !isNodeConfigured( configConn.farEnd.nodeId ) )
                continue;
            else
                    numConfigTrunks++;

            // Colwert FMPhysicalId_t to FMGpuOrSwitchId_t
            FMGpuOrSwitchId_t localGpuOrSwitchId, farGpuOrSwitchId;
            mGfm->getLWSwitchLWLinkDriverId(configConn.localEnd.nodeId, configConn.localEnd.lwswitchOrGpuId, localGpuOrSwitchId);
            mGfm->getLWSwitchLWLinkDriverId(configConn.farEnd.nodeId, configConn.farEnd.lwswitchOrGpuId, farGpuOrSwitchId);

            mGfm->mLWLinkDevRepo.getDeviceInfo(configConn.localEnd.nodeId, localGpuOrSwitchId, localDevInfo);
            mGfm->mLWLinkDevRepo.getDeviceInfo(configConn.farEnd.nodeId, farGpuOrSwitchId, farDevInfo);
            localDeviceName = localDevInfo.getDeviceName();
            farDeviceName = farDevInfo.getDeviceName();

            if( false == checkConnInDiscoveredConnections( configConn ) )
            {
                // configured connection was not found in list of discovered connections.
                // add configured connection to failed connections list
                FM_LOG_ERROR("LWLink trunk connection not found NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d "
                             "LWLinkIndex:%d <=======> NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d LWLinkIndex:%d",
                             configConn.localEnd.nodeId, localDeviceName.c_str(), configConn.localEnd.lwswitchOrGpuId,
                             mGfm->mpParser->getOsfpPortNumForLinkIndex(configConn.localEnd.portIndex), configConn.localEnd.portIndex,
                             configConn.farEnd.nodeId, farDeviceName.c_str(), configConn.farEnd.lwswitchOrGpuId,
                             mGfm->mpParser->getOsfpPortNumForLinkIndex(configConn.farEnd.portIndex), configConn.farEnd.portIndex);

                addFailedConnection( failedConnections, configConn );
                retVal = false;
            } else {
                FM_LOG_DEBUG("LWLink trunk connection found NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d "
                             "LWLinkIndex:%d <=======> NodeId:%d DeviceName:%s (PhysicalId:%d) OsfpPort:%d LWLinkIndex:%d",
                             configConn.localEnd.nodeId, localDeviceName.c_str(), configConn.localEnd.lwswitchOrGpuId,
                             mGfm->mpParser->getOsfpPortNumForLinkIndex(configConn.localEnd.portIndex), configConn.localEnd.portIndex,
                             configConn.farEnd.nodeId, farDeviceName.c_str(), configConn.farEnd.lwswitchOrGpuId,
                             mGfm->mpParser->getOsfpPortNumForLinkIndex(configConn.farEnd.portIndex), configConn.farEnd.portIndex);
            }
        }
    }
    return retVal;
}
#endif

void
FMTopologyValidator::checkLWSwitchLinkInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    // iterate for all the nodes
    for ( it = mGfm->mpParser->NodeCfg.begin(); it != mGfm->mpParser->NodeCfg.end(); it++ ) {
        NodeConfig *pNode = it->second;
        FMLWLinkDevInfoList devList;
        mGfm->mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        FMLWLinkDevInfoList::iterator jit;
        // iterate for all the device in this node
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            FMLWLinkDevInfo devInfo = (*jit);
            if (devInfo.getDeviceType () == lwlink_device_type_lwswitch) {
                // found an LWSwitch device, get its link initialization status
                checkLWSwitchDeviceLinkInitStatus(  pNode->nodeId, devInfo );
            }
        }
    }
}

void
FMTopologyValidator::checkLWSwitchDeviceLinkInitStatus(uint32_t nodeId,
                                                       FMLWLinkDevInfo &devInfo)
{
    if ( !devInfo.isAllLinksInitialized() ) {
        // found initialization failed links, check whether it is physically connected as some links
        // may not be wired ( DGX-2 bare metal have 2 unused links)
        std::list<uint32> initFailedLinks;
        std::list<uint32>::iterator it;
        devInfo.getInitFailedLinksIndex( initFailedLinks );
        for ( it = initFailedLinks.begin(); it != initFailedLinks.end(); it++ ) {
            uint32 linkIndex = (*it);
            FMLWSwitchInfo switchInfo;
            if (!getLWSwitchInfoByLWLinkDevInfo(nodeId, devInfo, switchInfo)) {
                FM_LOG_ERROR("no matching LWSwitch device for " NODE_ID_LOG_STR ":%d PCI bus id %s",
                             nodeId, devInfo.getDevicePciBusId());
                continue;
            }
            if (isSwitchPortConnected(nodeId, switchInfo.physicalId, linkIndex)) {
                // log initialization failures to syslog
                std::stringstream outStr;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
                uint32 osfpPort = mGfm->mpParser->getOsfpPortNumForLinkIndex(linkIndex);
                outStr << "LWLink initialization failed for NodeId:" << nodeId << " LWSwitch PCI bus id: " << switchInfo.pciInfo.busId
                       << "DeviceName:" << devInfo.getDeviceName() << " PhysicalId:" << switchInfo.physicalId;
                if( osfpPort != (uint32)-1 )
                    outStr << " OsfpPort:" << mGfm->mpParser->getOsfpPortNumForLinkIndex(linkIndex);
                outStr << " LWLinkIndex:" << linkIndex;
#else
                outStr << "LWLink initialization failed for LWSwitch PCI bus id: " << switchInfo.pciInfo.busId << "DeviceName:"
                       << devInfo.getDeviceName() << " PhysicalId:" << switchInfo.physicalId << " LWLinkIndex:" << linkIndex;
#endif
                FM_LOG_ERROR("%s", outStr.str().c_str());
                FM_SYSLOG_ERR("%s", outStr.str().c_str());
            }
        }
    }
}

void
FMTopologyValidator::checkGpuLinkInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    // iterate for all the nodes
    for ( it = mGfm->mpParser->NodeCfg.begin(); it != mGfm->mpParser->NodeCfg.end(); it++ ) {
        NodeConfig *pNode = it->second;
        FMLWLinkDevInfoList devList;
        mGfm->mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        FMLWLinkDevInfoList::iterator jit;
        // iterate for all the device in this node
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            FMLWLinkDevInfo devInfo = (*jit);
            if (devInfo.getDeviceType () == lwlink_device_type_gpu) {
                // found a GPU device, get its link initialization status
                checkGpuDeviceLinkInitStatus(  pNode->nodeId, devInfo );
            }
        }
    }
}

void
FMTopologyValidator::checkGpuDeviceLinkInitStatus(uint32_t nodeId,
                                                  FMLWLinkDevInfo &devInfo)
{
    if ( !devInfo.isAllLinksInitialized() ) {
        // for GPUs, all the enabled links should initialize success
        std::list<uint32> initFailedLinks;
        std::list<uint32>::iterator it;
        devInfo.getInitFailedLinksIndex( initFailedLinks );
        for ( it = initFailedLinks.begin(); it != initFailedLinks.end(); it++ ) {
            uint32 linkIndex = (*it);
            FMGpuInfo_t gpuInfo;
            getGpuInfoByLWLinkDevInfo(nodeId, devInfo, gpuInfo);
            // log initialization failures to syslog
            std::stringstream outStr;
            outStr << "LWLink initialization failed for " << NODE_ID_LOG_STR << ":" << nodeId << " GPU PCI bus id:" << gpuInfo.pciInfo.busId << " enumIndex:"
                   << gpuInfo.gpuIndex << " LWLinkIndex " << linkIndex;
            FM_LOG_ERROR("%s", outStr.str().c_str());
            FM_SYSLOG_ERR("%s", outStr.str().c_str());
        }
    }
}

uint32_t
FMTopologyValidator::getIntraNodeTrunkConnCount(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo)
{
    uint32_t connCount = 0;
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return count as zero
        return connCount;
    }

    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // count if the connection is trunk connection
        if (isLWLinkTrunkConnection(lwLinkConn)) {
            connCount++;
        }
    }

    FM_LOG_INFO("Number of LWSwitch intra-node trunk connections for " NODE_ID_LOG_STR " %d is %d", nodeId, connCount);

    return connCount;
}

uint32_t
FMTopologyValidator::getAccessConnCount(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo)
{
    uint32_t connCount = 0;
    LWLinkIntraConnMap tempIntraConnMap = lwLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);

    if (it == tempIntraConnMap.end()) {
        // no connection found for the node. return count as zero
        return connCount;
    }

    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator jit;
    for (jit = tempIntraConnList.begin(); jit != tempIntraConnList.end(); jit++) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;
        // count if the connection is not a trunk connection
        if (!isLWLinkTrunkConnection(lwLinkConn)) {
            connCount++;
        }
    }

    return connCount;

}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
uint32_t
FMTopologyValidator::getInterNodeTrunkConnCount(uint32_t nodeId)
{
    uint32_t connCount = 0;
    LWLinkInterNodeConns::iterator jit;
    LWLinkInterNodeConns interConnList = mGfm->mLWLinkConnRepo.getInterConnections();

    for ( jit = interConnList.begin(); jit != interConnList.end(); jit++ ) {
        // get the specific connection
        FMLWLinkDetailedConnInfo *lwLinkConn = *jit;

        // all the inter node connections should be trunk connections as of now,
        // so skipping explict isLWLinkTrunkConnection() check

        FMLWLinkEndPointInfo masterEndPoint = lwLinkConn->getMasterEndPointInfo();
        FMLWLinkEndPointInfo slaveEndPoint = lwLinkConn->getSlaveEndPointInfo();

        // count it if either master end or slave end of the connection belongs to
        // the specified node
        if ((masterEndPoint.nodeId == nodeId) || (slaveEndPoint.nodeId == nodeId)) {
            connCount++;
        }
    }

    FM_LOG_INFO("Number of LWSwitch inter-node trunk connections for " NODE_ID_LOG_STR " %d is %d", nodeId, connCount);

    return connCount;
}
#endif

// Check a specified switch port is connected
// A switch port is not connected due to the following reasons
// - a trunk port is not connected, due to
//    - not found in the topology, not all switch ports are used
//
// - an access port is not connected, due to
//    - not found in the topology, not all switch ports are used
//    - the absence of its peer GPU (physically not present, or manually excluded)
//    - peer GPU is present but has 0 enabledLinkMask, such as in the case of MIG enabled GPU
bool
FMTopologyValidator::isSwitchPortConnected(uint32_t nodeId, uint32_t physicalId, uint32_t linkIndex)
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
        // The switch port is not found in the topology file
        // no connection from the specified switch:port combination
        return false;
    }

    uint32_t nodeIdOtherEnd = 0;

    if (topoLWLinkConn.localEnd == switchEndPoint)
        nodeIdOtherEnd = topoLWLinkConn.farEnd.nodeId;
    else
        nodeIdOtherEnd = topoLWLinkConn.localEnd.nodeId;

    if (mGfm->isNodeDegraded(nodeIdOtherEnd)) {
        return false;
    }

    // if a trunk link is enabled in LWSwitch, it is expected to have a connection
    // in all the VM cases (only for 16GPU VM as of now)
    if (topoLWLinkConn.connType != ACCESS_PORT_GPU) {
        return true;
    }

    // We have an access connection from the switch:port combination.
    // Due to LWSwitch even-odd pair combination, some LWSwitch links may be enabled in certain VM combination,
    // without corresponding GPU side link enabled. So cross-check with GPU link mask to disable unwanted
    // error/warning messages.
    TopologyLWLinkConnEndPoint gpuEndPoint;
    // find the corresponding GPU endpoint side
    if (topoLWLinkConn.localEnd== switchEndPoint) 
        gpuEndPoint = topoLWLinkConn.farEnd;
    else
        gpuEndPoint = topoLWLinkConn.localEnd;

    FMGpuLWLinkConnMatrixMap::iterator it = mGfm->mGpuLWLinkConnMatrix.find(gpuEndPoint.nodeId);
    if (it == mGfm->mGpuLWLinkConnMatrix.end()) {
        // no GPU entry for the specified node
        return false;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for (jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++) {
        FMGpuLWLinkConnMatrix gpuConnInfo = *jit;

        if (gpuConnInfo.gpuPhyIndex == gpuEndPoint.lwswitchOrGpuId) {
            FMGpuInfo_t gpuInfo;
            if (mGfm->getGpuInfo(gpuConnInfo.uuid.bytes, gpuInfo) == false) {
                // cannot find the gpuInfo with specified uuid;
                return false;
            }

            // found the matching GPU, check whether corresponding link is enabled
            if (!(gpuInfo.enabledLinkMask & ((uint64)1 << gpuEndPoint.portIndex))) {
                // specified GPU link is not enabled, so don't count this connection
                // This is also true for MIG enabled GPU.
                return false;
            } else {
                // specified GPU link is enabled, treat this as a valid connection.
                return true;
            }
        }
    }

    // default cases like the GPU is not visible in the VM partition, ie FMGpuLWLinkConnMatrixMap
    // don't have the matching topology connection GPU.
    return false;
}

