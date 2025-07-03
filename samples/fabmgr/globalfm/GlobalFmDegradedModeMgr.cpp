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
#include <stdexcept>
#include <google/protobuf/text_format.h>
#include "fm_log.h"
#include "GlobalFmDegradedModeMgr.h"
#include "GFMHelper.h"

GlobalFmDegradedModeMgr::GlobalFmDegradedModeMgr(GlobalFabricManager *pGfm,
                                                 uint32_t accessLinkFailureMode,
                                                 uint32_t trunkLinkFailureMode,
                                                 uint32_t lwswitchFailureMode)
{
    mpGfm = pGfm;
    mAbortFm = false;

    mFailedSwitch.clear();
    mSwitchWithFailedTrunkPorts.clear();
    mSwitchWithFailedAccessPorts.clear();
    mSwitchWithFailedPorts.clear();
    mGpuWithFailedAccessPorts.clear();

    mDegradedSwitches.clear();
    mDegradedGpus.clear();
    mDisabledPartitions.clear();
    mLwlinkFailedDeviceMap.clear();
    mDegradedGpusByUuid.clear();
    mGpuWithAllLinksContain.clear();

    if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
        // Shared LWSwitch and vGPU Mode
        mGpuLinkFailureMode = (accessLinkFailureMode == 0) ?
                GPU_LINK_FAILURE_DISABLE_GPU :
                GPU_LINK_FAILURE_DISABLE_LWSWITCH;

        mLwswitchTrunkLinkFailureMode = (trunkLinkFailureMode == 0) ?
                LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION :
                LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH;

        mLwswitchFailureMode = (lwswitchFailureMode == 0) ?
                LWSWITCH_FAILURE_DISABLE_PARTITION :
                LWSWITCH_FAILURE_DISABLE_LWSWITCH;
    } else {
        // Baremetal or Full Pass-through mode
        mGpuLinkFailureMode = (accessLinkFailureMode == 0) ?
                GPU_LINK_FAILURE_DISABLE_GPU :
                GPU_LINK_FAILURE_DISABLE_LWSWITCH;

        mLwswitchTrunkLinkFailureMode = (trunkLinkFailureMode == 0) ?
                LWSWITCH_TRUNK_LINK_FAILURE_ABORT_FM :
                LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH;

        mLwswitchFailureMode = (lwswitchFailureMode == 0) ?
                LWSWITCH_FAILURE_ABORT_FM :
                LWSWITCH_FAILURE_DISABLE_LWSWITCH;
    }
};

GlobalFmDegradedModeMgr::~GlobalFmDegradedModeMgr()
{
    // nothing as of now
};

void
GlobalFmDegradedModeMgr::addFailedSwitch(uint32_t nodeId, uint32_t physicalId)
{
    SwitchKeyType key;
    key.nodeId = nodeId;
    key.physicalId = physicalId;

    // Switch with fatal errors
    mFailedSwitch.insert(key);
}

void
GlobalFmDegradedModeMgr::getAllFailedLwlinks(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator nit;
    // iterate for all the nodes
    for (nit = mpGfm->mpParser->NodeCfg.begin(); nit != mpGfm->mpParser->NodeCfg.end(); nit++) {

        NodeConfig *pNode = nit->second;
        FMLWLinkDevInfoList devList;
        mpGfm->mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );

        FMLWLinkDevInfoList::iterator dit;
        // iterate for all the device in this node
        for ( dit = devList.begin(); dit != devList.end(); dit++ ) {
            FMLWLinkDevInfo devInfo = (*dit);

            if (devInfo.getDeviceType () == lwlink_device_type_lwswitch) {
                getFailedLwlinksFromSwitch(pNode->nodeId, devInfo);
            }

            if (devInfo.getDeviceType () == lwlink_device_type_gpu) {
                getFailedLwlinksFromGpu(pNode->nodeId, devInfo);
            }
        }
    }
}

void
GlobalFmDegradedModeMgr::removeLinkFromMissingConnLinkList(uint32 linkIndex, std::list<uint32> &missingConnLinkIndex) 
{
    std::list<uint32>::iterator it;
    for (it = missingConnLinkIndex.begin(); it != missingConnLinkIndex.end(); it++) {
        if (*it == linkIndex) {
            missingConnLinkIndex.erase(it);
            break;
        }
    }
}

void
GlobalFmDegradedModeMgr::updateSwitchFailedTrunkLinkInfoByPeerSwitch(SwitchKeyType switchKey, SwitchKeyType peerSwitchKey,
                                                                     std::list<uint32_t> &peerFailedTrunkPortList)
{
    std::map <SwitchKeyType, FailedPortList>::iterator it = mSwitchWithFailedPorts.find(switchKey);
    if (it == mSwitchWithFailedPorts.end()) {
        // The switch does not have a failed port list yet
        // add one
        FailedPortList portList;
        portList.clear();
        mSwitchWithFailedPorts.insert(std::make_pair(switchKey, portList));
        it = mSwitchWithFailedPorts.find(switchKey);
    }
    FailedPortList &failedPortList = it->second;

    std::list<uint32_t>::iterator jit;
    for (jit = peerFailedTrunkPortList.begin(); jit != peerFailedTrunkPortList.end(); jit++) {

        PortKeyType trunkPortKey;
        trunkPortKey.nodeId = peerSwitchKey.nodeId;
        trunkPortKey.physicalId = peerSwitchKey.physicalId;
        trunkPortKey.portIndex = *jit;

        // Test if the trunk port on the peer switch is connected to this switch
        // In multi node systems, not all trunk ports on one switch are connect to the
        // same peer switch
        uint32_t localPortNum = 0;
        if (!mpGfm->mpParser->isTrunkConnectedToSwitch(trunkPortKey, switchKey, localPortNum)) {
            // The peer trunk port is not connected to this switch
            continue;
        }

        // add this end of the failed trunk link to this switch's failedportlist
        FailedPortList::iterator fit;
        bool portAlreadyOnList = false;
        for (fit = failedPortList.begin(); fit != failedPortList.end(); fit++) {
            uint32_t portNum = (*fit);
            if (portNum == localPortNum) {
                // this trunk port is already on the failed port list
                portAlreadyOnList = true;
                break;
            }
        }

        if (!portAlreadyOnList) {
            // add the port to failedPortList
            failedPortList.push_back(localPortNum);
        }
    }
}

void
GlobalFmDegradedModeMgr::getFailedLwlinksFromSwitch(uint32_t nodeId, FMLWLinkDevInfo &devInfo)
{
    // Find the switch info
    FMLWSwitchInfoMap::iterator jit = mpGfm->mLwswitchInfoMap.find(nodeId);
    if (jit == mpGfm->mLwswitchInfoMap.end()) {
        return;
    }

    FMLWSwitchInfoList switchList = jit->second;
    FMLWSwitchInfoList::iterator sit;
    FMLWSwitchInfo *switchInfo = NULL;
    lwlink_pci_dev_info lwlinkPciInfo = devInfo.getDevicePCIInfo();

    for (sit = switchList.begin(); sit != switchList.end(); sit++) {
        FMLWSwitchInfo *tempInfo = &(*sit);

        if ((tempInfo->pciInfo.domain == lwlinkPciInfo.domain) &&
             (tempInfo->pciInfo.bus == lwlinkPciInfo.bus) &&
             (tempInfo->pciInfo.device == lwlinkPciInfo.device) &&
             (tempInfo->pciInfo.function == lwlinkPciInfo.function)) {
            switchInfo = tempInfo;
            break;
        }
    }

    //If switch is excluded then no need to check any further, return 
    FMExcludedLWSwitchInfoMap::iterator bjit = mpGfm->mExcludedLwswitchInfoMap.find(nodeId);
    if (bjit == mpGfm->mExcludedLwswitchInfoMap.end()) {
        return;
    }

    FMExcludedLWSwitchInfoList excludedSwitchList = bjit->second;
    FMExcludedLWSwitchInfoList::iterator bsit;
    lwlinkPciInfo = devInfo.getDevicePCIInfo();

    for (bsit = excludedSwitchList.begin(); bsit != excludedSwitchList.end(); bsit++) {
        FMExcludedLWSwitchInfo_t *tempInfo = &(*bsit);

        if ((tempInfo->pciInfo.domain == lwlinkPciInfo.domain) &&
            (tempInfo->pciInfo.bus == lwlinkPciInfo.bus) &&
            (tempInfo->pciInfo.device == lwlinkPciInfo.device) &&
            (tempInfo->pciInfo.function == lwlinkPciInfo.function)) {
            return;
        }
    }

    if (!switchInfo) {
        FM_LOG_ERROR("Failed to find " NODE_ID_LOG_STR " %d LWSwitch with PCI %x.%x.%x.%x",
                     nodeId, lwlinkPciInfo.domain, lwlinkPciInfo.bus,
                     lwlinkPciInfo.device, lwlinkPciInfo.function);
        return;
    }

    SwitchKeyType switchKey;
    switchKey.nodeId = nodeId;
    switchKey.physicalId = switchInfo->physicalId;
 
    ConnectedSwitchesInfoMap connectedSwitches;
    ConnectedSwitchesInfoMap::iterator switchIt;
 
    mpGfm->mpParser->getConnectedSwitches( switchKey, connectedSwitches );
 
    for (switchIt = connectedSwitches.begin(); switchIt != connectedSwitches.end(); switchIt++) {
        SwitchKeyType pairKey = switchIt->first;
        mSwitchPairs[switchKey].push_back(pairKey);
        std::map <SwitchKeyType, FailedPortList>::iterator fit = mSwitchWithFailedTrunkPorts.find(pairKey);
        if (fit != mSwitchWithFailedTrunkPorts.end()) {
            // The peer switch is already processed
            // Still need to add failure trunk ports on this switch so that
            // fmGetLwlinkFailedDevices API will return both ends of failed
            // trunk ports
            FailedPortList failedTrunkPortList= fit->second;
            updateSwitchFailedTrunkLinkInfoByPeerSwitch(switchKey, pairKey, failedTrunkPortList);
            return;
        }
    }
 
    PortKeyType portKey;
    portKey.nodeId = nodeId;
    portKey.physicalId = switchInfo->physicalId;

    FailedPortList failedTrunkPorts;
    FailedPortList failedAccessPorts;
    FailedPortList allFailedPorts;
    std::list<uint32> initFailedLinks;
    std::list<uint32> missingConnLinkIndex; 
    std::list<uint32>::iterator it;

    failedTrunkPorts.clear();
    failedAccessPorts.clear();
    allFailedPorts.clear();

    // all non active links are considered failed
    devInfo.getNonActiveLinksIndex( initFailedLinks, missingConnLinkIndex );

    for ( it = initFailedLinks.begin(); it != initFailedLinks.end(); it++ ) {
        portKey.portIndex = (*it);

        std::map <PortKeyType, lwswitch::switchPortInfo *>::iterator pit;
        pit = mpGfm->mpParser->portInfo.find(portKey);
        if (pit == mpGfm->mpParser->portInfo.end()) {
            FM_LOG_DEBUG("Failed to find port on " NODE_ID_LOG_STR " %d LWSwitch physicalId %d, portNum %d",
                         nodeId, portKey.physicalId, portKey.portIndex);
            // The port might not be connected.
            continue;
        }

        if (!mpGfm->mTopoValidator->isSwitchPortConnected(nodeId, portKey.physicalId,
                                                          portKey.portIndex)) {
            // This switch port is in the topology but not connected, due to
            // Trunk port is not connected, or
            // GPU might be excluded, or
            // GPU might be in MIG mode
            // remove any connection from the missingConnLinkIndex list, since we
            // dont need to log missing connections that arent actually meant to be connected,
            // like missing connections due to MIG enabled GPUs
            removeLinkFromMissingConnLinkList(portKey.portIndex, missingConnLinkIndex);
            continue;
        }

        lwswitch::switchPortInfo *portInfo = pit->second;
        if (portInfo->has_config()) {
            switchPortConfig portConfig = portInfo->config();
            allFailedPorts.push_back(portKey.portIndex);

            if (portConfig.has_type() &&
                (portConfig.type() == TRUNK_PORT_SWITCH)) {
                failedTrunkPorts.push_back(portKey.portIndex);
            } else {
                failedAccessPorts.push_back(portKey.portIndex);
            }
        }
    }

    // log information about all missing links
    mpGfm->mTopoValidator->logNonDetectedLWLinkConns(missingConnLinkIndex, switchKey.nodeId, switchKey.physicalId, lwlink_device_type_lwswitch);

    if (failedTrunkPorts.size() > 0) {
        mSwitchWithFailedTrunkPorts.insert(std::make_pair(switchKey, failedTrunkPorts));
    }

    if (failedAccessPorts.size() > 0) {
        mSwitchWithFailedAccessPorts.insert(std::make_pair(switchKey, failedAccessPorts));
    }

    if (allFailedPorts.size() > 0) {
        mSwitchWithFailedPorts.insert(std::make_pair(switchKey, allFailedPorts));
    }
}

void
GlobalFmDegradedModeMgr::getFailedLwlinksFromGpu(uint32_t nodeId, FMLWLinkDevInfo &devInfo)
{
    GpuKeyType key;
    key.nodeId = nodeId;

    FMGpuInfo_t gpuInfo;
    mpGfm->mTopoValidator->getGpuInfoByLWLinkDevInfo(nodeId, devInfo, gpuInfo);
    if (!mpGfm->getGpuPhysicalId(nodeId, gpuInfo.uuid.bytes, key.physicalId)) {
        // the device uuid is not found
        FM_LOG_ERROR("Failed to find GPU with " NODE_ID_LOG_STR " %d UUID %s.", nodeId, gpuInfo.uuid.bytes);
        return;
    }

    FailedPortList failedPorts;
    std::list<uint32> missingConnLinkIndex;
    // all non active links are considered failed, unless they are in contain mode
    // links could enter contain mode even if neighboring links have failed
    // such links will be not be considered as failed links
    devInfo.getFailedNonContainLinksIndex(failedPorts, missingConnLinkIndex);
    // log information about missing links
    mpGfm->mTopoValidator->logNonDetectedLWLinkConns(missingConnLinkIndex, key.nodeId, key.physicalId, lwlink_device_type_gpu);

    if (devInfo.isAllLinksInContain()) {
        mGpuWithAllLinksContain.insert(key);
    }

    if (failedPorts.size() > 0) {
        mGpuWithFailedAccessPorts.insert(std::make_pair(key, failedPorts));
    }
}

bool
GlobalFmDegradedModeMgr::isSwitchDegraded(uint32_t nodeId, uint32_t physicalId,
                                          lwswitch::SwitchDegradedReason &reason)
{
    SwitchKeyType key;
    key.nodeId = nodeId;
    key.physicalId = physicalId;

    DegradedLWSwitchMap::iterator it;
    it = mDegradedSwitches.find(key);

    if (it != mDegradedSwitches.end()) {
        reason = it->second;
        return true;
    }

    return false;
}

bool
GlobalFmDegradedModeMgr::isGpuDegraded(uint32_t nodeId, uint32_t physicalId,
                                       lwswitch::GpuDegradedReason &reason)
{
    GpuKeyType key;
    key.nodeId = nodeId;
    key.physicalId = physicalId;

    DegradedGpuMap::iterator it;
    it = mDegradedGpus.find(key);

    if (it != mDegradedGpus.end()) {
        reason = it->second;
        return true;
    }

    return false;
}

bool
GlobalFmDegradedModeMgr::isPartitionDisabled(uint32_t nodeId, uint32_t partitionId)
{
    PartitionKeyType key;
    key.nodeId = nodeId;
    key.partitionId = partitionId;

    PartitionSet::iterator it = mDisabledPartitions.find(key);

    if (it == mDisabledPartitions.end()) {
        return false;
    } else {
        return true;
    }
}

/*
    This function makes sure that we dont consider connected pairs of lwswitches
    that are excluded as muliple excluded lwswitches. In such a case
    we need not modify the option to fall back to a case LWSWITCH_FAILURE_DISABLE_PARTITION 
*/
bool
GlobalFmDegradedModeMgr::isMultipleLWSwitchesExcluded(uint32_t nodeId, FMExcludedLWSwitchInfoList &excludedInfoList)
{
    FMExcludedLWSwitchInfoList::iterator it;
    std::set<SwitchKeyType> excludedSwitchSet;
    int numSwitchesExcluded = 0;

    for (it = excludedInfoList.begin(); it != excludedInfoList.end(); it++) {
        SwitchKeyType key;
        FMExcludedLWSwitchInfo_t excludedSwitch = *it;
        key.physicalId = excludedSwitch.physicalId;
        key.nodeId = nodeId;

        if (excludedSwitchSet.find(key) != excludedSwitchSet.end()) {
            // already added this switch to the set, need not count it
            continue;
        }

        numSwitchesExcluded++;
        excludedSwitchSet.insert(key);

        ConnectedSwitchesInfoMap connectedSwitches;
        ConnectedSwitchesInfoMap::iterator sit;
        mpGfm->getExcludedConnectedSwitches( key, connectedSwitches );

        for (sit = connectedSwitches.begin(); sit != connectedSwitches.end(); sit++) {
            SwitchKeyType peerSwitchKey = sit->first;
            excludedSwitchSet.insert(peerSwitchKey);
        }
    }

    // as connected LWSwitch pairs are counted as 1 switch, we return true
    // if numSwitchesExcluded is greater than 1
    return numSwitchesExcluded > 1;
}

void
GlobalFmDegradedModeMgr::processExcludedSwitches()
{ 
    //TODO: change for multi node
    uint32_t nodeId = 0;
    FMExcludedLWSwitchInfoMap excludedLwswitchInfo = mpGfm->getExcludedLwswitchInfoMap();
    FMExcludedLWSwitchInfoList excludedInfoList = excludedLwswitchInfo[nodeId];
    FMExcludedLWSwitchInfoList::iterator it;

    if (excludedInfoList.size() == 0) {
        return;
    }
    FM_LOG_DEBUG("Excluded switches excludedInfoList.size = %lu", excludedInfoList.size());

    // if all excluded switches are because of FM degrading them
    // then just return, need not degrade those switches
    for (it = excludedInfoList.begin(); it != excludedInfoList.end(); it++) {
        FMExcludedLWSwitchInfo_t excludedSwitch = *it;
        if (excludedSwitch.excludedReason == lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED) {
            break;
        }
    }

    if (it == excludedInfoList.end()) {
        return;
    }

    if (mLwswitchFailureMode == LWSWITCH_FAILURE_ABORT_FM) {
        // option is set to abort FM, nothing need to be done
        mAbortFm = true;
        FM_LOG_ERROR("excluded LWSwitch detected and degraded mode configuration is set to abort Fabric Manager");
        FM_SYSLOG_ERR("excluded LWSwitch detected and degraded mode configuration is set to abort Fabric Manager");
        return;
    }

    // Adjust option when more than one LWSwitches is excluded
    if (isMultipleLWSwitchesExcluded(nodeId, excludedInfoList) && 
        mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_LWSWITCH ) 
    {
        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
            // shared fabric mode
            // fallback to LWSWITCH_FAILURE_DISABLE_PARTITION
            mLwswitchFailureMode = LWSWITCH_FAILURE_DISABLE_PARTITION;
            FM_LOG_INFO("more than one LWSwitch is excluded, disable partitions using excluded LWSwitches.");
        } else {
            // in bare metal or full pass through
            // Abort
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            mLwswitchFailureMode = LWSWITCH_FAILURE_DISABLE_LWSWITCH;
            FM_LOG_INFO("setting LWSwitch Failure Mode to disable switches in cases where multiple switches are excluded for multi node cases");
#else
            mLwswitchFailureMode = LWSWITCH_FAILURE_ABORT_FM;
            mAbortFm = true;
            FM_LOG_INFO("more than one LWSwitch is excluded, aborting fabric manager and leaving LWLinks and the system uninitialized.");
            throw std::runtime_error("more than one LWSwitch is excluded, aborting fabric manager and leaving LWLinks and the system uninitialized.");
            // option is to abort FM, nothing need to be done
            return;
#endif
        }
    }

    for (it = excludedInfoList.begin(); it != excludedInfoList.end(); it++) {
        SwitchKeyType key;
        FMExcludedLWSwitchInfo_t excludedSwitch = *it;
        key.physicalId = excludedSwitch.physicalId;
        key.nodeId = nodeId;
        degradeOneSwitch(key, lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED);
    }
}

void
GlobalFmDegradedModeMgr::processFailures(void)
{
    // 1. gather all LWLink failures.
    getAllFailedLwlinks();

    // 2. process excluded switches
    processExcludedSwitches();

    // 3. process switch failures
    // if a switch is already degraded, there is no need to look at trunk link failures on it
    processSwitchFailures();

    // 4. process trunk link failures
    processTrunkLinkFailures();

    // 5. process access link failures
    processAccessLinkFailures();

    if (mAbortFm) {
        throw std::runtime_error("GPU or LWSwitch failure oclwrred and degraded mode configuration evaluates to abort Fabric Manager");
    }

    // 6. turn off all Lwlinks on degraded switches and GPUs
    turnOffLwlinksOnDegradedDev(mpGfm->mLinkTrainIntf,
                                mpGfm->mLWLinkConnRepo,
                                mpGfm->mLWLinkDevRepo);

    // 7. disabled all affected partitions if there is any
    mpGfm->mpParser->disablePartitions(mDisabledPartitions);

    // 8. send degraded info
    sendDegradedInfoToAllNodes();

    // 9. set up mDegradedDeviceList for API query, use nodeId 0 for now
    uint32_t nodeId = 0;
    populateLwlinkFailedDeviceMap(nodeId);

    // 10. build map of degraded gpus for lookup based on uuid(required for GPU reset)
    buildDegradedGpuUuidLookup();
}

void
GlobalFmDegradedModeMgr::sendDegradedInfoToAllNodes()
{
    if (mDegradedSwitches.size() > 0) {
        sendAllDegradedLwswitchInfo();
    }
}

bool
GlobalFmDegradedModeMgr::isAnyDeviceDegraded()
{
    if (mDegradedSwitches.size() > 0 || mDegradedGpus.size() > 0) {
        return true;
    }

    return false;
}

void
GlobalFmDegradedModeMgr::degradeOneSwitch(SwitchKeyType &key, lwswitch::SwitchDegradedReason reason)
{
    lwswitch::SwitchDegradedReason tempReason;
    if (isSwitchDegraded(key.nodeId, key.physicalId, tempReason) == true) {
        // The switch is already degraded
        FM_LOG_DEBUG("LWSwitch " NODE_ID_LOG_STR " %d physical Id %d is already degraded with reason %d.",
                     key.nodeId, key.physicalId, tempReason);
        return;
    }

    // find all shared LWSwitch partitions that are using this switch
    PartitionSet partitions;
    if (reason != lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED) {
        mpGfm->mpParser->getSharedLWSwitchPartitionsWithSwitch( key, partitions);
    }
    else {
        mpGfm->getPartitionSetForExcludedSwitch(key, partitions);
    }

    // the check for the degraded reason is done to ensure that the corresponding degraded mode
    // option is taken into account only when necessary. For example, a case where we 
    // excluded a switch, and we have option for trunk link failure set to 1, and option for 
    // excludeding set to 0, it would go inside this if condition if we did not include the reason
    // in the check. 
    if  ( (mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_LWSWITCH && reason == lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED)
        || (mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_LWSWITCH && reason == lwswitch::LWSWITCH_FAILURE)
        || (mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH && reason == lwswitch::TRUNK_LINK_FAILURE)
        || (mGpuLinkFailureMode == GPU_LINK_FAILURE_DISABLE_LWSWITCH && reason == lwswitch::ACCESS_LINK_FAILURE)
        ) 
    {
        mDegradedSwitches.insert(std::make_pair(key, reason));
        FMPciInfo_t pciInfo;
        if (reason != lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED) {
            mpGfm->getLWSwitchPciBdf(key.nodeId, key.physicalId, pciInfo);
        }
        else {
            mpGfm->getExcludedLWSwitchPciBdf(key.nodeId, key.physicalId, pciInfo);
        }

        FM_LOG_INFO("degrading switch with " NODE_ID_LOG_STR " %d physical id %d, pci bus id %s due to %s", 
                    key.nodeId, key.physicalId, pciInfo.busId, 
                    getReasonAsString(reason));

        /* The map, modifiedPartitions consists of each modified partition with the number of 
           degraded links in that partition */ 
        std::map<uint32_t, int> modifiedPartitions;
        std::map<uint32_t, int>::iterator sit;
        // update GPU enabled link mask on the shared LWSwitch partitions in shared fabric mode
        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
            mpGfm->mpParser->removeSwitchFromSharedLWSwitchPartitions(key, modifiedPartitions);
        }
        
        // all active links on this degraded switch will be trained to off by turnOffLwlinksOnDegradedDev

        // degrade all the connected peer switches
        // on a single node system, there should be 0 or 1 connected switch
        ConnectedSwitchesInfoMap connectedSwitches;
        ConnectedSwitchesInfoMap::iterator it;

        if (reason != lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED) {
            mpGfm->mpParser->getConnectedSwitches( key, connectedSwitches );
        }
        else {
            mpGfm->getExcludedConnectedSwitches( key, connectedSwitches );
        }

        for (it = connectedSwitches.begin(); it != connectedSwitches.end(); it++) {

            SwitchKeyType switchKey = it->first;
            lwswitch::SwitchDegradedReason peerSwitchReason;
            getPeerReason(reason, peerSwitchReason);

            mDegradedSwitches.insert(std::make_pair(switchKey, peerSwitchReason));
            mpGfm->getLWSwitchPciBdf(switchKey.nodeId, switchKey.physicalId, pciInfo);
            FM_LOG_INFO("degrading switch with " NODE_ID_LOG_STR " %d physical id %d, pci bus id %s due to %s", 
                        switchKey.nodeId, switchKey.physicalId, pciInfo.busId, 
                        getReasonAsString(peerSwitchReason));

            // update GPU enabled link mask on the shared LWSwitch partitions in shared fabric mode
            modifiedPartitions.clear();
            if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
                mpGfm->mpParser->removeSwitchFromSharedLWSwitchPartitions(switchKey, modifiedPartitions);
            }
            // all active links on this degraded switch will be trained to off by turnOffLwlinksOnDegradedDev
        }

    } else if ((mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_PARTITION && reason == lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED)
               || (mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_PARTITION && reason == lwswitch::LWSWITCH_FAILURE)
               || (mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION && reason == lwswitch::TRUNK_LINK_FAILURE)
              ) 
    {
        // add all partitions that are using this switch to mDisabledPartitions
        PartitionSet::iterator pit;
        for (pit = partitions.begin(); pit != partitions.end(); pit++) {
            PartitionKeyType key = *pit;
            mDisabledPartitions.insert(key);

            FM_LOG_INFO("disabling shared LWSwitch multitenancy fabric partition with " NODE_ID_LOG_STR " %d and partitionid %d due to %s.",
                         key.nodeId, key.partitionId, getReasonAsString(reason));

        }

        // verify disable cross base board partitions that are using degraded peer switches
    }
}

void
GlobalFmDegradedModeMgr::degradeOneGpu(GpuKeyType &key, lwswitch::GpuDegradedReason reason)
{
    lwswitch::GpuDegradedReason tempReason;
    if (isGpuDegraded(key.nodeId, key.physicalId, tempReason) == true) {
        // The GPU is already degraded
        FM_LOG_DEBUG("GPU  " NODE_ID_LOG_STR " %d physical Id %d is already degraded with reason %d.",
                     key.nodeId, key.physicalId, tempReason);
        return;
    }

    if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
        // find all shared LWSwitch partitions that are using this GPU
        PartitionSet partitions;
        mpGfm->mpParser->getSharedLWSwitchPartitionsWithGpu( key, partitions);

        // add all partitions that are using this switch to mDisabledPartitions
        PartitionSet::iterator it;
        for (it = partitions.begin(); it != partitions.end(); it++) {
            PartitionKeyType key = *it;

            if ((reason == lwswitch::LWLINK_FAILURE) &&
                (mpGfm->mpParser->getNumGpusInPartition(key) == 1)) {
                // GPU with lwlink failure can still participate in 1 GPU partitions
                continue;
            }

            mDisabledPartitions.insert(key);
            FM_LOG_INFO("disabling shared LWSwitch multitenancy fabric partition with " NODE_ID_LOG_STR " %d and partitionid %d due to a degraded GPU.",
                         key.nodeId, key.partitionId);
        }
    }

    mDegradedGpus.insert(std::make_pair(key, reason));
    uint32_t gpuEnumIndex;
    FMPciInfo_t pciInfo; 
    mpGfm->getGpuEnumIndex(key.nodeId, key.physicalId, gpuEnumIndex);
    mpGfm->getGpuPciBdf(key.nodeId, gpuEnumIndex, pciInfo);
    FM_LOG_INFO("degrading GPU with " NODE_ID_LOG_STR " %d pci bus id %s physical id %d due to access lwlink failure", key.nodeId, pciInfo.busId, key.physicalId);
}

void
GlobalFmDegradedModeMgr::processSwitchFailures(void)
{
    if (mAbortFm == true) {
        // option is already set to abort FM, nothing need to be done
        return;
    }

    if (mFailedSwitch.size() == 0) {
        // no failed switch
        return;
    }

    if (mLwswitchFailureMode == LWSWITCH_FAILURE_ABORT_FM) {
        // option is set to abort FM, nothing need to be done
        mAbortFm = true;
        FM_LOG_ERROR("LWSwitch failure detected and degraded mode configuration set to abort Fabric Manager");
        FM_SYSLOG_ERR("LWSwitch failure detected and degraded mode configuration set to abort Fabric Manager");
        return;
    }

    // Adjust option when more than one LWSwitches failed
    // and LWSWITCH_FAILURE_MODE=1
    if (mFailedSwitch.size() > 1 && mLwswitchFailureMode == LWSWITCH_FAILURE_DISABLE_LWSWITCH) {

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
            // shared fabric mode
            // fallback to LWSWITCH_FAILURE_DISABLE_PARTITION
            mLwswitchFailureMode = LWSWITCH_FAILURE_DISABLE_PARTITION;

            FM_LOG_INFO("more than one LWSwitch failed, disable partitions using failed LWSwitches.");

        } else {
            // in bare metal or full pass through
            // Abort
            mLwswitchFailureMode = LWSWITCH_FAILURE_ABORT_FM;
            mAbortFm = true;
            FM_LOG_INFO("more than one LWSwitch failed, aborting fabric manager and leaving LWLinks and the system uninitialized.");
            throw std::runtime_error("more than one LWSwitch failed, aborting fabric manager and leaving LWLinks and the system uninitialized.");
            // option is to abort FM, nothing need to be done
            return;
        }
    }

    std::set <SwitchKeyType>::iterator it;
    for (it = mFailedSwitch.begin(); it != mFailedSwitch.end(); ++it) {
        SwitchKeyType key = *it;
        degradeOneSwitch(key, lwswitch::LWSWITCH_FAILURE);
    }
}

void
GlobalFmDegradedModeMgr::processTrunkLinkFailures(void)
{
    if (mAbortFm == true) {
        // option is already set to abort FM, nothing need to be done
        return;
    }

    if (mSwitchWithFailedTrunkPorts.size() == 0) {
        // no failed trunk links
        return;
    }

    if (mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_ABORT_FM) {
        // option is to abort FM, nothing need to be done
        mAbortFm = true;
        FM_LOG_ERROR("Trunk LWLink failures detected and degrade mode configuration set to abort Fabric Manager");
        FM_SYSLOG_ERR("Trunk LWLink failures detected and degrade mode configuration set to abort Fabric Manager");
        return;
    }

    // Adjust option when more than one LWSwitches have trunk link failures
    if ( mSwitchWithFailedTrunkPorts.size() > 1 && mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH) {
        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
            // shared fabric mode
            // fallback to LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION
            mLwswitchTrunkLinkFailureMode = LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION;

            FM_LOG_INFO("More than one LWSwitch failed, disable partitions using failed trunk LWLinks.");

        } else {
            // in bare metal or full pass through
            // Abort
            mLwswitchTrunkLinkFailureMode = LWSWITCH_TRUNK_LINK_FAILURE_ABORT_FM;
            mAbortFm = true;
            FM_LOG_INFO("More than one LWSwitches have trunk LWLink failed, set to abort Fabric manager.");
            throw std::runtime_error("More than one LWSwitches have trunk LWLink failed, set to abort Fabric manager.");
            // option is to abort FM, nothing need to be done
            return;
        }
    }

    std::map <SwitchKeyType, FailedPortList>::iterator it;
    for (it = mSwitchWithFailedTrunkPorts.begin(); it != mSwitchWithFailedTrunkPorts.end(); it++) {

        if (mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION) {
            // find all partitions that are using trunk links and disable
            PartitionSet partitions;
            PartitionSet::iterator pit;
            mpGfm->mpParser->getSharedLWSwitchPartitionsWithTrunkLinks( 0, partitions );

            for (pit = partitions.begin(); pit != partitions.end(); pit++) {
                PartitionKeyType key = *pit;
                mDisabledPartitions.insert(key);
                FM_LOG_INFO("disabling shared LWSwitch multitenancy fabric partition with " NODE_ID_LOG_STR " %d and partitionid %d due to a LWSwitch trunk failure.",
                             key.nodeId, key.partitionId);
            }

        } else if (mLwswitchTrunkLinkFailureMode == LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH) {
            SwitchKeyType key = it->first;
            degradeOneSwitch(key, lwswitch::TRUNK_LINK_FAILURE);
        }
    }
}

void
GlobalFmDegradedModeMgr::processAccessLinkFailures(void)
{
    if (mAbortFm == true) {
        // option is already set to abort FM, nothing need to be done
        return;
    }

    if (mGpuWithFailedAccessPorts.size() == 0) {
        // no failed access links
        return;
    }

    /*
        if multiple gpus fail and they are connected to different switches
        we should abort
    */
    if (mGpuWithFailedAccessPorts.size() > 1 && mSwitchWithFailedAccessPorts.size() > 1) {
        //set option to abort
        mAbortFm = true;
        FM_LOG_INFO("Multiple GPUs connected to multiple switches failing. set to abort Fabric manager ");
        throw std::runtime_error("More than one LWSwitches have trunk LWLink failed, set to abort Fabric manager.");
        return;
    }

    SwitchKeyType switchKey;
    std::set <SwitchKeyType> switchList;
    std::map <GpuKeyType, FailedPortList>::iterator it;

    for (it = mGpuWithFailedAccessPorts.begin(); it != mGpuWithFailedAccessPorts.end(); it++) {

        switchList.clear();
        GpuKeyType key = it->first;
        FailedPortList &portList = it->second;

        FailedPortList::iterator pit;
        uint32_t switchPhysicalId;
 
        switchKey.nodeId = key.nodeId;
 
        //find all switches connected to this particular gpu with failed port
        for (pit = portList.begin(); pit != portList.end(); pit++) {
            // find the switch end point that is connect to this GPU port
            TopologyLWLinkConnEndPoint gpuEndPoint;
            gpuEndPoint.nodeId = key.nodeId;
            gpuEndPoint.lwswitchOrGpuId = key.physicalId;
            gpuEndPoint.portIndex = *pit;
 
            TopologyLWLinkConn topoLWLinkConn;
            if (!mpGfm->mTopoValidator->getTopologyLWLinkConnByGpuEndPoint(gpuEndPoint, topoLWLinkConn)) {
                // could not find the switch end point
                FM_LOG_ERROR("Failed to find switch end port with GPU " NODE_ID_LOG_STR " %d, physicalId %d, port %d",
                             gpuEndPoint.nodeId, gpuEndPoint.lwswitchOrGpuId, gpuEndPoint.portIndex );
                continue;
            }
 
            // found the switch port that is connected to failed gpu port
            switchKey.physicalId = topoLWLinkConn.localEnd.lwswitchOrGpuId;
            lwswitch::SwitchDegradedReason reason;
            FMLWSwitchInfo switchInfoUnused;
            if (mpGfm->isLwswitchExcluded(switchKey.nodeId, switchKey.physicalId) || 
                (isSwitchDegraded(switchKey.nodeId, switchKey.physicalId, reason))) {
                // the gpu port is connected to a excluded switch
                // or the gpu port is connected to a switch degraded due to trunk link failure/peer excluded
                // no need to count the switch
            } else if(!mpGfm->getLWSwitchInfo(switchKey.nodeId, switchKey.physicalId, switchInfoUnused)) {
                // these access link failures are due to links connected
                // to switch that are removed/unbinded
                // these could also be due to the reason that switch is not part of a full pass through VM
                // TODO: test with full passthrough VMs
                if (mFailedSwitch.find(switchKey) == mFailedSwitch.end())
                    mFailedSwitch.insert(switchKey);
            }
            else {
                switchList.insert(switchKey);
            }
        }

        if (switchList.size() == 0) {
            // all GPU link failures are due to excluded switch/already degraded switch
            // or due to switches that have been unbinded/removed. These are process seperately
            // move on to the next GPU
            continue;
        }
 
        // adjust the option if more than one access port failed
        // and if ACCESS_LINK_FAILURE_MODE is 1
        if ( portList.size() > 1 && mGpuLinkFailureMode == GPU_LINK_FAILURE_DISABLE_LWSWITCH) {
            if (switchList.size() == 1) {
                // multiple failed GPU links are connected to the same switch
                if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
                    FM_LOG_INFO("More than one access LWLink failed on the same GPU,"
                                "set to isolate the GPU.");
                    mGpuLinkFailureMode = GPU_LINK_FAILURE_DISABLE_GPU;
                } else {
                    // in bare metal use GPU_LINK_FAILURE_DISABLE_SWITCH
                    FM_LOG_INFO("More than one LWLinks failed on the GPU connecting to same LWSwitch, "
                                "set to disable the LWSwitch.");
                    mGpuLinkFailureMode = GPU_LINK_FAILURE_DISABLE_LWSWITCH;
                }
            } else {
                // multiple failed GPU links are connected to different switches
                if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU)) {
                    FM_LOG_INFO("More than one access LWLink failed on the same GPU,"
                                "set to isolate the GPU.");
                    mGpuLinkFailureMode = GPU_LINK_FAILURE_DISABLE_GPU;

                } else {
                    // in bare metal fall back to GPU_LINK_FAILURE_DISABLE_GPU
                    FM_LOG_INFO("More than one access LWLink failed on the same GPU connecting different LWSwitches,"
                                "set to isolate the GPU.");
                    mGpuLinkFailureMode = GPU_LINK_FAILURE_DISABLE_GPU;

                }
            }
        }

        if (mGpuLinkFailureMode == GPU_LINK_FAILURE_DISABLE_LWSWITCH && mGpuWithAllLinksContain.find(key) != mGpuWithAllLinksContain.end()) {
            // this is a case where only one access link failure oclwrs, but it results in all links 
            // entering contain mode. If switch is degraded, alert user to reset GPU to train
            // links other than failed link to high
            if ((mpGfm->getFabricMode() == FM_MODE_BAREMETAL)) {
                uint32_t gpuEnumIndex;
                FMPciInfo_t pciInfo; 
                mpGfm->getGpuEnumIndex(key.nodeId, key.physicalId, gpuEnumIndex);
                mpGfm->getGpuPciBdf(key.nodeId, gpuEnumIndex, pciInfo);
                FM_LOG_ERROR("GPU with " NODE_ID_LOG_STR " %d physical id %d, pci bus id %s experienced an LWLink error and requires " 
                             "reset before launching LWCA jobs. Please refer to your system user guide for GPU reset instructions", 
                             key.nodeId, key.physicalId, pciInfo.busId);
                FM_SYSLOG_ERR("GPU with " NODE_ID_LOG_STR " %d physical id %d, pci bus id %s experienced an LWLink error and requires " 
                             "reset before launching LWCA jobs. Please refer to your system user guide for GPU reset instructions", 
                             key.nodeId, key.physicalId, pciInfo.busId);
            }
        }

        if (mGpuLinkFailureMode == GPU_LINK_FAILURE_DISABLE_GPU) {
            degradeOneGpu(key, lwswitch::LWLINK_FAILURE);
        }

        if (mGpuLinkFailureMode == GPU_LINK_FAILURE_DISABLE_LWSWITCH) {
            std::set <SwitchKeyType>::iterator sit;
            for (sit = switchList.begin(); sit != switchList.end(); sit++) {
                switchKey = *sit;
                degradeOneSwitch(switchKey, lwswitch::ACCESS_LINK_FAILURE);
            }
        }
    }

    // if there are any switches that are removed, we need to take
    // care of those as well.
    if (mFailedSwitch.size() > 0) {
        processSwitchFailures();
    }
}

void
GlobalFmDegradedModeMgr::handleDegradeGpuInfoAckMsg(lwswitch::fmMessage *pFmMessage)
{
    if ( !pFmMessage->has_gpudegradedinfoack() ) {
        // empty response
        FM_LOG_ERROR("received an empty response for request to set GPU degraded state.");
        return;
    }

    const lwswitch::gpuDegradedInfoAck &respMsg = pFmMessage->gpudegradedinfoack();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.response_size() == 0 ) {
        // no instance response
        FM_LOG_ERROR("received no instance response for request to set GPU degraded state");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ ) {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) ) {
            FM_LOG_ERROR("request to set GPU degraded state message failed for " NODE_ID_LOG_STR " %d physical id %d with error %d.",
                        nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.status());

            // TODO, what needs to be done if OOB access failed
        }
    }
}

void
GlobalFmDegradedModeMgr::handleDegradeLwswitchInfoAckMsg(lwswitch::fmMessage *pFmMessage)
{
    if ( !pFmMessage->has_switchdegradedinfoack() ) {
        // empty response
        FM_LOG_ERROR("received an empty response for request to set LWSwitch degraded state.");
        return;
    }

    const lwswitch::switchDegradedInfoAck &respMsg = pFmMessage->switchdegradedinfoack();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.response_size() == 0 ) {
        // no instance response
        FM_LOG_ERROR("received no instance response for request to set LWSwitch degraded state.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ ) {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) ) {
            FM_LOG_ERROR("request to set LWSwitch degraded state message failed for " NODE_ID_LOG_STR " %d physical id %d with error %d.",
                        nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1, 
                        instanceResponse.status());

            // TODO, what needs to be done if OOB access failed
        }
    }
}

void
GlobalFmDegradedModeMgr::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    FM_LOG_DEBUG("message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_DEGRADED_GPU_INFO_ACK:
        dumpMessage(pFmMessage);
        handleDegradeGpuInfoAckMsg( pFmMessage );
        break;

    case lwswitch::FM_DEGRADED_LWSWITCH_INFO_ACK:
        dumpMessage(pFmMessage);
        handleDegradeLwswitchInfoAckMsg( pFmMessage );
        break;

    default:
        FM_LOG_ERROR("unknown message type %d received in degraded mode message handler", pFmMessage->type());
        break;
    }
}

void
GlobalFmDegradedModeMgr::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    FM_LOG_DEBUG("%s", msgText.c_str());
#endif
}

void
GlobalFmDegradedModeMgr::sendDegradedGpuInfo(uint32_t nodeId)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuDegradedInfo *pDegradedGpuInfo = new lwswitch::gpuDegradedInfo();

    // fill in the degraded GPUs on this node
    DegradedGpuMap::iterator it;

    for (it = mDegradedGpus.begin(); it != mDegradedGpus.end(); it++) {
        GpuKeyType key = it->first;
        lwswitch::GpuDegradedReason reason = it->second;

        if (key.nodeId != nodeId)
            continue;

        lwswitch::gpuDegraded *pGpuDegraded = pDegradedGpuInfo->add_gpuinfo();
        pGpuDegraded->set_physicalid(key.physicalId);
        pGpuDegraded->set_reason(reason);
    }

    // send the messages to LFM
    pMessage->set_type(lwswitch::FM_DEGRADED_GPU_INFO);
    pMessage->set_allocated_gpudegradedinfo(pDegradedGpuInfo);

    rc = SendMessageToLfm(nodeId, pMessage);
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("request to send degraded GPU info to " NODE_ID_LOG_STR " %d failed with error %d.", nodeId, rc);

    }
}

void
GlobalFmDegradedModeMgr::sendDegradedLwswitchInfo(uint32_t nodeId)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::switchDegradedInfo *pDegradedSwitchInfo = new lwswitch::switchDegradedInfo();
    int numDegradedSwitches = 0;

    // fill in the degraded switches on this node
    DegradedLWSwitchMap::iterator it;

    for (it = mDegradedSwitches.begin(); it != mDegradedSwitches.end(); it++) {
        SwitchKeyType key = it->first;
        lwswitch::SwitchDegradedReason reason = it->second;

        if (key.nodeId != nodeId)
            continue;
 
        // if switch is not in context, ie, not found/unbinded/excluded,
        // then we need not send degraded message to LFM
        FMLWSwitchInfo switchInfoNotUsed;
        if (!mpGfm->getLWSwitchInfo(nodeId, key.physicalId, switchInfoNotUsed)) {
            continue;
        }

        lwswitch::switchDegraded *pSwitchDegraded = pDegradedSwitchInfo->add_switchinfo();
        pSwitchDegraded->set_physicalid(key.physicalId);
        pSwitchDegraded->set_reason(reason);
        numDegradedSwitches++;
    }

    if (numDegradedSwitches < 1) {
        delete pDegradedSwitchInfo;
        delete pMessage;
        return;
    }

    // send the messages to LFM
    pMessage->set_type(lwswitch::FM_DEGRADED_LWSWITCH_INFO);
    pMessage->set_allocated_switchdegradedinfo(pDegradedSwitchInfo);

    rc = SendMessageToLfm(nodeId, pMessage);
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("request to send degraded LWSwitch info to " NODE_ID_LOG_STR " %d failed with error %d.", nodeId, rc);
    }
}

void
GlobalFmDegradedModeMgr::sendAllDegradedGpuInfo(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator nodeIt;

    for (nodeIt = mpGfm->mpParser->NodeCfg.begin(); nodeIt != mpGfm->mpParser->NodeCfg.end(); nodeIt++) {
        NodeConfig *pNode = nodeIt->second;
        sendDegradedGpuInfo(pNode->nodeId);
    }
}

void
GlobalFmDegradedModeMgr::sendAllDegradedLwswitchInfo(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator nodeIt;

    for (nodeIt = mpGfm->mpParser->NodeCfg.begin(); nodeIt != mpGfm->mpParser->NodeCfg.end(); nodeIt++) {
        NodeConfig *pNode = nodeIt->second;
        sendDegradedLwswitchInfo(pNode->nodeId);
    }
}

FMIntReturn_t
GlobalFmDegradedModeMgr::SendMessageToLfm(uint32_t nodeId, lwswitch::fmMessage *pFmMessage)
{
    FMIntReturn_t ret;
    FMIntReturn_t rc = FM_INT_ST_OK;

    if (!pFmMessage) {
        FM_LOG_DEBUG("Invalid message to " NODE_ID_LOG_STR " %d.", nodeId);
        return FM_INT_ST_MSG_SEND_ERR;
    }

    int id = mpGfm->getControlMessageRequestId(nodeId);
    pFmMessage->set_requestid(id);

    // Do not track message
    ret = mpGfm->SendMessageToLfm(nodeId, pFmMessage, false);
    if ( ret != FM_INT_ST_OK ) {
        FM_LOG_ERROR("request to send socket message to " NODE_ID_LOG_STR " %d failed with error %d", nodeId, ret);
        rc = FM_INT_ST_MSG_SEND_ERR;
    }

    delete pFmMessage;
    return rc;
}

int
GlobalFmDegradedModeMgr::turnOffLwlinksOnDegradedDev(GlobalFMLWLinkIntf *linkTrainIntf,
                                                     GlobalFMLWLinkConnRepo &linkConnRepo,
                                                     GlobalFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint64, FMLWLinkDetailedConnInfo*> requestIds;
    std::map<uint64, FMLWLinkDetailedConnInfo*>::iterator reqIt;
    LWLinkIntraConnMap::iterator it;
    int retVal = 0;
    LWLinkIntraConnMap& intraConnMap = linkConnRepo.getIntraConnections();

    FM_LOG_INFO( "training all LWLink connections on degraded devices to off" );

    // first send train request
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfoList &connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;
        for (jit = connList.begin(); jit != connList.end(); jit++) {
            uint64 requestId = 0;
            FMLWLinkReq linkReq = {{0}};
            lwswitch::SwitchDegradedReason sReason;
            lwswitch::GpuDegradedReason gReason;

            FMLWLinkDetailedConnInfo *conn = *jit;

            // skip if the connection is not trained to active or is not in contain state
            if (!conn->isConnTrainedToActive() && !conn->isConnInContainState()) {
                continue;
            }

            // fill master end information of the connection
            FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
            linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
            linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
            // fill slave end information of the connection
            FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();
            linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
            linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
            linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;

            FMLWLinkDevInfo masterDevInfo, slaveDevInfo;
            FMLWSwitchInfo masterSwitchInfo, slaveSwitchInfo;

            linkDevRepo.getDeviceInfo( masterEnd.nodeId, masterEnd.gpuOrSwitchId, masterDevInfo );
            linkDevRepo.getDeviceInfo( slaveEnd.nodeId, slaveEnd.gpuOrSwitchId, slaveDevInfo );

            if ( masterDevInfo.getDeviceType() == lwlink_device_type_lwswitch ) {
                if ( mpGfm->mTopoValidator->getLWSwitchInfoByLWLinkDevInfo(masterEnd.nodeId,
                                                                           masterDevInfo,
                                                                           masterSwitchInfo) &&
                     isSwitchDegraded(masterEnd.nodeId, masterSwitchInfo.physicalId, sReason) ) {
                    // the masterEnd switch is degraded
                    // fill the train to state
                    linkReq.connTrainReq.trainTo = LWLINK_TRAIN_TO_OFF;
                    linkTrainIntf->sendConnTrainReq( linkReq, requestId );
                    if (requestId) {
                        requestIds.insert( std::make_pair(requestId, conn) );
                    }
                    continue;
                }
            }

            if ( slaveDevInfo.getDeviceType() == lwlink_device_type_lwswitch ) {
                if ( mpGfm->mTopoValidator->getLWSwitchInfoByLWLinkDevInfo(slaveEnd.nodeId,
                                                                           slaveDevInfo,
                                                                           slaveSwitchInfo) &&
                     isSwitchDegraded(slaveEnd.nodeId, slaveSwitchInfo.physicalId, sReason) ) {
                    // the slaveEnd switch is degraded
                    linkReq.connTrainReq.trainTo = LWLINK_TRAIN_TO_OFF;
                    linkTrainIntf->sendConnTrainReq( linkReq, requestId );
                    if (requestId) {
                        requestIds.insert( std::make_pair(requestId, conn) );
                    }
                    continue;
                }
            }

            uint32_t physicalId;
            FMGpuInfo_t masterGpuInfo, slaveGpuInfo;

            if ( masterDevInfo.getDeviceType() == lwlink_device_type_gpu ) {
                if ( mpGfm->mTopoValidator->getGpuInfoByLWLinkDevInfo(masterEnd.nodeId,
                                                                      masterDevInfo,
                                                                      masterGpuInfo) &&
                     mpGfm->getGpuPhysicalId(masterEnd.nodeId, masterGpuInfo.uuid.bytes,
                                             physicalId) &&
                     isGpuDegraded(masterEnd.nodeId, physicalId, gReason) ) {

                    // the masterEnd GPU is degraded
                    linkReq.connTrainReq.trainTo = LWLINK_TRAIN_TO_OFF;
                    linkTrainIntf->sendConnTrainReq( linkReq, requestId );
                    if (requestId) {
                        requestIds.insert( std::make_pair(requestId, conn) );
                    }
                    continue;
                }
            }

            if ( slaveDevInfo.getDeviceType() == lwlink_device_type_gpu ) {
                if ( mpGfm->mTopoValidator->getGpuInfoByLWLinkDevInfo(slaveEnd.nodeId,
                                                                      slaveDevInfo,
                                                                      slaveGpuInfo) &&
                     mpGfm->getGpuPhysicalId(slaveEnd.nodeId, slaveGpuInfo.uuid.bytes,
                                             physicalId) &&
                     isGpuDegraded(slaveEnd.nodeId, physicalId, gReason) ) {

                    // the masterEnd GPU is degraded
                    linkReq.connTrainReq.trainTo = LWLINK_TRAIN_TO_OFF;
                    linkTrainIntf->sendConnTrainReq( linkReq, requestId );
                    if (requestId) {
                        requestIds.insert( std::make_pair(requestId, conn) );
                    }
                    continue;
                }
            }
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                // update the training/link status information to the connection
                FMLWLinkDetailedConnInfo *connInfo = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    std::stringstream outStr;
                    connInfo->dumpConnInfo( &outStr, mpGfm, linkDevRepo);
                    retVal = reqResult.status;
                    FM_LOG_ERROR("following LWLink connection training to off failed with error: %s\n %s",
                                 FMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str());
                    // log connection training failures to syslog
                    FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
                }
                // update the training/link status information to the connection/device repo
                GFMHelper::updateDeviceAndConnLinkState( linkDevRepo, connInfo, reqResult.connTrainResp );
                requestIds.erase( reqIt++ );
            } else {
                ++reqIt;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            lwosThreadYield();
        }
    }

    return 0;
}

void
GlobalFmDegradedModeMgr::getPeerReason(lwswitch::SwitchDegradedReason reason, lwswitch::SwitchDegradedReason &peerSwitchReason)
{
    switch(reason) {
        case lwswitch::LWSWITCH_FAILURE:
            peerSwitchReason = lwswitch::LWSWITCH_PEER_FAILURE;
            break;
        case lwswitch::TRUNK_LINK_FAILURE:
            peerSwitchReason = lwswitch::PEER_LWSWITCH_DEGRADED_TRUNK_LINK;
            break;
        case lwswitch::ACCESS_LINK_FAILURE:
            peerSwitchReason = lwswitch::PEER_LWSWITCH_DEGRADED_ACCESS_LINK;
            break;
        case lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED:
            peerSwitchReason = lwswitch::PEER_DEGRADE_EXPLICITLY_EXCLUDED_SWITCH;
            break;
        default:
            break;
    }
}

const char*
GlobalFmDegradedModeMgr::getReasonAsString(lwswitch::SwitchDegradedReason reason)
{
    switch (reason) {
        case lwswitch::LWSWITCH_FAILURE:
            return "LWSwitch failure";

        case lwswitch::LWSWITCH_PEER_FAILURE:
            return "LWSwitch peer failure";

        case lwswitch::TRUNK_LINK_FAILURE:
            return "trunk LWLink failure";

        case lwswitch::ACCESS_LINK_FAILURE:
            return "access LWLink failure";

        case lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED:
            return "LWSwitch explicitly excluded";

        case lwswitch::PEER_LWSWITCH_DEGRADED_TRUNK_LINK:
            return "peer LWSwitch trunk LWLink failure";

        case lwswitch::PEER_LWSWITCH_DEGRADED_ACCESS_LINK:
            return "peer LWSwitch access LWLink failure";

        case lwswitch::PEER_DEGRADE_EXPLICITLY_EXCLUDED_SWITCH:
            return "peer LWSwitch explicitly excluded";

        default:
            return "unknown reason";
    }
}

void
GlobalFmDegradedModeMgr::getLwlinkFailedGpus(uint32_t nodeId, uint32_t &numGpus, fmLwlinkFailedDeviceInfo_t gpuInfo[])
{
    int i, j;

    // get all failed gpu ports
    std::map <GpuKeyType, FailedPortList>::iterator it;

    i = 0;
    for (it = mGpuWithFailedAccessPorts.begin(); it != mGpuWithFailedAccessPorts.end(); it++) {
        GpuKeyType gpukey = it->first;
        if (gpukey.nodeId != nodeId) {
            continue;
        }

        FailedPortList gpuPortList = it->second;

        // get the uuid and pciInfo from GFM
        mpGfm->getGpuUuid(nodeId, gpukey.physicalId, gpuInfo[i].uuid);
        FMPciInfo_t pciInfo = {0};
        if (mpGfm->getGpuPciBdf(nodeId, gpuInfo[i].uuid, pciInfo)) {
            memcpy(&gpuInfo[i].pciBusId, pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
        }

        // fill in all failed ports
        gpuInfo[i].numPorts = gpuPortList.size();
        FailedPortList::iterator pit;

        for (pit = gpuPortList.begin(), j = 0; pit != gpuPortList.end(); pit++, j++) {
            gpuInfo[i].portNum[j] = (*pit);
        }

        i++;
    }

    numGpus = i;
}

void
GlobalFmDegradedModeMgr::getLwlinkFailedSwitches(uint32_t nodeId, uint32_t &numSwitches, fmLwlinkFailedDeviceInfo_t switchInfo[])
{
    int i, j;

    // get all failed switch ports
    std::map <SwitchKeyType, FailedPortList>::iterator it;
    i = 0;

    for (it = mSwitchWithFailedPorts.begin(); it != mSwitchWithFailedPorts.end(); it++) {
        SwitchKeyType switchKey = it->first;
        if (switchKey.nodeId != nodeId) {
            continue;
        }

        FailedPortList switchPortList = it->second;

        // fill in switch uuid and pciInfo
        FMLWSwitchInfo fmSwitchInfo;
        if (mpGfm->getLWSwitchInfo(nodeId, switchKey.physicalId, fmSwitchInfo)) {
            memcpy(switchInfo[i].uuid, fmSwitchInfo.uuid.bytes, FM_UUID_BUFFER_SIZE);
            memcpy(&switchInfo[i].pciBusId, fmSwitchInfo.pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
        }

        // fill in all failed ports
        switchInfo[i].numPorts = switchPortList.size();
        FailedPortList::iterator pit;

        for (pit = switchPortList.begin(), j = 0; pit != switchPortList.end(); pit++, j++) {
            switchInfo[i].portNum[j] = (*pit);
        }

        i++;
    }

    numSwitches = i;
}

void
GlobalFmDegradedModeMgr::populateLwlinkFailedDeviceMap(uint32_t nodeId)
{
    fmLwlinkFailedDevices_t devList = {0};

    getLwlinkFailedGpus(nodeId, devList.numGpus, devList.gpuInfo);
    getLwlinkFailedSwitches(nodeId, devList.numSwitches, devList.switchInfo);

    if ((devList.numGpus > 0) || (devList.numSwitches > 0)) {
        mLwlinkFailedDeviceMap.insert(make_pair(nodeId, devList));
    }
}

void
GlobalFmDegradedModeMgr::getLwlinkFailedDevices(uint32_t nodeId, fmLwlinkFailedDevices_t *devList)
{
    std::map <uint32_t, fmLwlinkFailedDevices_t>::iterator it = mLwlinkFailedDeviceMap.find(nodeId);

    if (it != mLwlinkFailedDeviceMap.end()) {
        fmLwlinkFailedDevices_t *degradedDevicesList = &(it->second);
        memcpy(devList, degradedDevicesList, sizeof(fmLwlinkFailedDevices_t));
    }
}

void 
GlobalFmDegradedModeMgr::buildDegradedGpuUuidLookup()
{
    DegradedGpuMap::iterator it;
    for (it = mDegradedGpus.begin(); it != mDegradedGpus.end(); it++) {
        GpuKeyType key = it->first;
        FMUuid_t uuid = {{0}};
        char tempUuid[FM_UUID_BUFFER_SIZE] = {0};
        mpGfm->getGpuUuid(key.nodeId, key.physicalId, tempUuid);
        strncpy(uuid.bytes, tempUuid, FM_UUID_BUFFER_SIZE - 1);
        mDegradedGpusByUuid.insert(uuid);
    }
}

bool
GlobalFmDegradedModeMgr::isGpuDegraded(char *gpuUuid) 
{
    FMUuid_t uuid = {{0}};
    strncpy(uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE - 1);
    if (mDegradedGpusByUuid.find(uuid) != mDegradedGpusByUuid.end()) {
        return true;
    }

    return false;
}

int
GlobalFmDegradedModeMgr::getNumPairsSwitchDegraded()
{
    DegradedLWSwitchMap::iterator it;
    int numDegradedSwitchPairs = 0;
    std::set<SwitchKeyType> degradedSwitchSet;

    for (it = mDegradedSwitches.begin(); it != mDegradedSwitches.end(); it++) {
        SwitchKeyType key = it->first;

        if (degradedSwitchSet.find(key) != degradedSwitchSet.end()) {
            continue;
        }

        numDegradedSwitchPairs++;
        degradedSwitchSet.insert(key);

        ConnectedSwitchesInfoMap connectedSwitches;
        ConnectedSwitchesInfoMap::iterator switchIt;
     
        mpGfm->mpParser->getConnectedSwitches( key, connectedSwitches );
     
        for (switchIt = connectedSwitches.begin(); switchIt != connectedSwitches.end(); switchIt++) {
            SwitchKeyType pairKey = switchIt->first;
            degradedSwitchSet.insert(pairKey);
        }
    }

    return numDegradedSwitchPairs;
}

// this function computes the number of access links degraded for gpu due to switch degradation
int
GlobalFmDegradedModeMgr::getNumDegradedLinksForGpu()
{
    if (mDegradedSwitches.size() == 0) 
        return 0;

    int numSwitchPairsDegraded = getNumPairsSwitchDegraded();

    // multiplied by 2 for ampere based systems
    // TODO: revisit for future architectures, multi-node
    return numSwitchPairsDegraded * 2;
}
