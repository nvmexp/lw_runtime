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

#include "fm_log.h"
#include "FMCommonTypes.h"
#include "GFMHelper.h"
#include "GlobalFabricManager.h"
#include "GFMFabricPartitionMgr.h"
#include "FMAutoLock.h"

GlobalFMFabricPartitionMgr::GlobalFMFabricPartitionMgr(GlobalFabricManager *pGfm)
{
    mGfm = pGfm;
    mSupportedPartitions.clear();
    memset(&mUnsupportedPartitionList, 0, sizeof(fmUnsupportedFabricPartitionList_t));
    lwosInitializeCriticalSection(&mLock);

    mInitDone = false;
    mResetPendingGpus.clear();
}

GlobalFMFabricPartitionMgr::~GlobalFMFabricPartitionMgr()
{
    lwosDeleteCriticalSection(&mLock);
}

/*************************************************************************************
 * From the static shared LWSwitch partition information specified in the topology file,
 * build the available partition information and corresponding GPU information.
 *
 * This mapping is created once all the LWLink training is completed with all the
 * available GPUs and LWSwitches (as part of initialization)
 * 
 * This should only be called when Partition Mgr is started for the first time,
 * The partition mapping is loaded from the save state at restart.
 *
 **************************************************************************************/
bool
GlobalFMFabricPartitionMgr::buildPartitionMappings()
{
    if (mGfm->isFabricModeRestart() == true) {
        FM_LOG_ERROR("build fabric partition is called during restart mode");
        return false;
    }

    // lwrrently look for only node 0.
    // Note: re-visit during multi-node support
    std::map <NodeKeyType, NodeConfig *>::iterator it = mGfm->mpParser->NodeCfg.begin();
    if(it == mGfm->mpParser->NodeCfg.end()) {
        FM_LOG_WARNING("valid fabric config is not found while building fabric partition mapping");
        return false;
    }

    // iterate through all the partitions specified in the topology file
    NodeConfig *pNodeCfg = it->second;
    for (int idx = 0; idx < pNodeCfg->partitionInfo.sharedlwswitchinfo_size(); idx++) {
        const sharedLWSwitchPartitionInfo &topoPartInfo = pNodeCfg->partitionInfo.sharedlwswitchinfo(idx);
        if (!mapTopologyPartition(pNodeCfg->nodeId, topoPartInfo)) {
            FM_LOG_WARNING("failed to create fabric partition mapping for " NODE_ID_LOG_STR " %d partition id %d.",
                            pNodeCfg->nodeId, topoPartInfo.partitionid());
            continue;
        }
    }

    // also build the unsupported partition list
    populateUnsupportedPartitions();

    mInitDone = true;
    return true;
}

/*
 * get partition mapping to mpHaState protobuf
 */
bool
GlobalFMFabricPartitionMgr::getSharedFabricHaState(fabricmanagerHA::sharedFabricPartiontionInfo &haState)
{
    FM_LOG_DEBUG("getting shared fabric partition list information");

    if (mSupportedPartitions.size() == 0) {
        // mSupportedPartitions is not ready yet
        FM_LOG_ERROR("shared fabric partition list information is not ready yet.");
        return false;
    }

    fabricmanagerHA::infoHeader *haHeader = new fabricmanagerHA::infoHeader;
    haState.set_allocated_header(haHeader);
    fabricmanagerHA::nodePartitionInfo *haNodeInfo = haState.add_nodelist();

    // set to the lwrrently supported version
    haHeader->set_version(FABRIC_PARTITION_HA_STATE_VER);

    // TODO, Need to revisit for multi-node
    haNodeInfo->set_nodeid(0);

    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &partInfo = *it;

        FM_LOG_DEBUG("saving partition Id %d and its state %d", partInfo.partitionId, partInfo.partitionState);

        fabricmanagerHA::partitionInfo *haPartInfo = haNodeInfo->add_partitionlist();
        haPartInfo->set_partitionid(partInfo.partitionId);
        haPartInfo->set_partitionstate(partInfo.partitionState);
        haPartInfo->set_trunkconncount(partInfo.trunkConnCount);

        // save all GPUs in the partition
        PartitionGpuInfoList::iterator gpuIt;
        for (gpuIt = partInfo.gpuInfo.begin(); gpuIt != partInfo.gpuInfo.end(); gpuIt++) {
            PartitionGpuInfo &partGpuInfo = (*gpuIt);

            fabricmanagerHA::partitionGpuInfo *gpuHaInfo = haPartInfo->add_gpulist();
            gpuHaInfo->set_physicalid(partGpuInfo.physicalId);
            gpuHaInfo->set_uuid(partGpuInfo.uuid);
            gpuHaInfo->set_numenabledlinks(partGpuInfo.numEnabledLinks);
            gpuHaInfo->set_enabledlinkmask(partGpuInfo.enabledLinkMask);
            gpuHaInfo->set_numlinksavailable(partGpuInfo.numLinksAvailable);
            gpuHaInfo->set_linklineratembps(partGpuInfo.linkLineRateMBps);
            gpuHaInfo->set_discoveredlinkmask(partGpuInfo.discoveredLinkMask);

            fabricmanagerHA::pciInfo *pciInfo = new fabricmanagerHA::pciInfo;
            pciInfo->set_domain(partGpuInfo.dynamicInfo.pciInfo.domain);
            pciInfo->set_bus(partGpuInfo.dynamicInfo.pciInfo.bus);
            pciInfo->set_device(partGpuInfo.dynamicInfo.pciInfo.device);
            pciInfo->set_function(partGpuInfo.dynamicInfo.pciInfo.function);
            pciInfo->set_busid(partGpuInfo.dynamicInfo.pciInfo.busId);
            gpuHaInfo->set_allocated_pciinfo(pciInfo);

            // For vGPU mode, save gfid specific information
            gpuHaInfo->set_gfid(partGpuInfo.dynamicInfo.gfidInfo.gfid);
            gpuHaInfo->set_gfidmask(partGpuInfo.dynamicInfo.gfidInfo.gfidMask);
       }

       // save all LWSwitches in the partition
       PartitionSwitchInfoList::iterator switchIt;
       for (switchIt = partInfo.switchInfo.begin();
            switchIt != partInfo.switchInfo.end();
            switchIt++) {
           PartitionSwitchInfo &partSwitchInfo = (*switchIt);

           fabricmanagerHA::partitionSwitchInfo *switchHaInfo = haPartInfo->add_switchlist();
           switchHaInfo->set_physicalid(partSwitchInfo.physicalId);
           switchHaInfo->set_archtype(partSwitchInfo.archType);
           switchHaInfo->set_numenabledlinks(partSwitchInfo.numEnabledLinks);
           switchHaInfo->set_enabledlinkmask(partSwitchInfo.enabledLinkMask);
       }
    }

    return true;
}

/*
 * Validate a previously saved partition mapping
 */
bool
GlobalFMFabricPartitionMgr::validateSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState)
{
    // version check
    if (savedHaState.header().version() != FABRIC_PARTITION_HA_STATE_VER) {
        // TODO: process message upgrade/down grade
    }

    // number of node check
    if (savedHaState.nodelist_size() != (int)mGfm->mpParser->NodeCfg.size()) {
        // TODO: comment out the syslog on multi node before multi node is supported

        FM_LOG_DEBUG("Saved node size %d is different from topology node size %d",
                    (int)savedHaState.nodelist_size(), (int)mGfm->mpParser->NodeCfg.size());

        // FM_SYSLOG_ERR("Saved node size %d is different from topology node size %d",
        //              (int)savedHaState.nodelist_size(), (int)mGfm->mpParser->NodeCfg.size());
        return false;
    }

    // number of partition check
    // TODO: use nodeId 0, revisit for multi node
    NodeConfig *pNodeCfg = NULL;
    for ( std::map <NodeKeyType, NodeConfig *>::iterator nodeit = mGfm->mpParser->NodeCfg.begin();
          nodeit != mGfm->mpParser->NodeCfg.end();
          nodeit++)
    {
        pNodeCfg = nodeit->second;
        if (pNodeCfg->nodeId == 0) {
            break;
        }
    }
    if (pNodeCfg == NULL) {
        FM_LOG_ERROR("required fabric partition config is not found in the topology file while rebuilding fabric manager states");
        return false;
    }

    const fabricmanagerHA::nodePartitionInfo *pNodeHaInfo = NULL;
    for (int n = 0; n < savedHaState.nodelist_size(); n++) {
        if ((savedHaState.nodelist(n).has_nodeid()) &&
            (savedHaState.nodelist(n).nodeid() == 0)) {
            pNodeHaInfo = &savedHaState.nodelist(n);
        }
    }

    if (pNodeHaInfo == NULL) {
        FM_LOG_ERROR("fabric partition information is missing in saved state file");
        return false;
    }

    // check if the number of saved partitions are more than the number
    // defined for the node
    if (pNodeCfg->partitionInfo.sharedlwswitchinfo_size() < pNodeHaInfo->partitionlist_size()) {
        std::stringstream outStr;
        outStr << "number of fabric partitions in saved state file is more than the number of partitions specified in topology file"
               << " saved state count:" << pNodeHaInfo->partitionlist_size() << " topology partition count:"
               << pNodeCfg->partitionInfo.sharedlwswitchinfo_size();
        FM_LOG_ERROR("%s", outStr.str().c_str());
        FM_SYSLOG_ERR("%s", outStr.str().c_str());
        return false;
    }

    return true;
}

/*
 * From a saved partition states, restore shared partition mapping states.
 * The saved partition states need to be validated before calling this.
 */
bool
GlobalFMFabricPartitionMgr::loadSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState)
{
    FM_LOG_DEBUG("loading shared fabric partition list information");

    for (int n = 0; n < savedHaState.nodelist_size(); n++) {
        const fabricmanagerHA::nodePartitionInfo &nodeHaInfo = savedHaState.nodelist(n);

        for (int i = 0;  i < nodeHaInfo.partitionlist_size(); i++) {
            PartitionInfo partInfo;
            PartitionSwitchInfoList switchInfoList;
            PartitionGpuInfoList gpuInfoList;

            switchInfoList.clear();
            gpuInfoList.clear();

            const fabricmanagerHA::partitionInfo &partHaInfo = nodeHaInfo.partitionlist(i);

            // load each GPU information
            for ( int gpuIdx = 0; gpuIdx < partHaInfo.gpulist_size(); gpuIdx++ ) {
                PartitionGpuInfo gpuInfo;
                const fabricmanagerHA::partitionGpuInfo &gpuHaInfo = partHaInfo.gpulist(gpuIdx);
                gpuInfo.physicalId = gpuHaInfo.physicalid();
                gpuInfo.numEnabledLinks = gpuHaInfo.numenabledlinks();
                gpuInfo.enabledLinkMask = gpuHaInfo.enabledlinkmask();
                memset(gpuInfo.uuid, 0, sizeof(gpuInfo.uuid));
                strncpy(gpuInfo.uuid, gpuHaInfo.uuid().c_str(), FM_UUID_BUFFER_SIZE - 1);
                gpuInfo.numLinksAvailable = gpuHaInfo.numlinksavailable();
                gpuInfo.linkLineRateMBps = gpuHaInfo.linklineratembps();
                gpuInfo.discoveredLinkMask = gpuHaInfo.discoveredlinkmask();

                if (gpuHaInfo.has_pciinfo()) {
                    gpuInfo.dynamicInfo.pciInfo.domain = gpuHaInfo.pciinfo().domain();
                    gpuInfo.dynamicInfo.pciInfo.bus = gpuHaInfo.pciinfo().bus();
                    gpuInfo.dynamicInfo.pciInfo.device = gpuHaInfo.pciinfo().device();
                    gpuInfo.dynamicInfo.pciInfo.function = gpuHaInfo.pciinfo().function();
                    strncpy(gpuInfo.dynamicInfo.pciInfo.busId, gpuHaInfo.pciinfo().busid().c_str(),
                            FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE - 1);
                }

                // For vGPU mode, load gfid specific information
                if (gpuHaInfo.has_gfid()) {
                    gpuInfo.dynamicInfo.gfidInfo.gfid = gpuHaInfo.gfid();
                    gpuInfo.dynamicInfo.gfidInfo.gfidMask = gpuHaInfo.gfidmask();
                }

                gpuInfoList.push_back(gpuInfo);
            }

            // load each Switch information
            for ( int switchIdx = 0; switchIdx < partHaInfo.switchlist_size(); switchIdx++ ) {
                PartitionSwitchInfo switchInfo;
                const fabricmanagerHA::partitionSwitchInfo &swHaInfo = partHaInfo.switchlist(switchIdx);
                switchInfo.physicalId = swHaInfo.physicalid();
                switchInfo.archType = swHaInfo.archtype();
                switchInfo.numEnabledLinks = swHaInfo.numenabledlinks();
                switchInfo.enabledLinkMask = swHaInfo.enabledlinkmask();
                switchInfoList.push_back(switchInfo);
            }

            partInfo.partitionId = partHaInfo.partitionid();

            // set all previously activated partition state to sync pending state
            if (partHaInfo.has_partitionstate() && (partHaInfo.partitionstate() == PARTITION_IN_ACTIVE_STATE)) {
                partInfo.partitionState = PARTITION_IN_SYNC_PENDING_STATE;
            } else {
                partInfo.partitionState = PARTITION_IN_DEACTIVE_STATE;

                //
                // TODO: Remove the following check once we enhance state saving logic
                // for shared lwswitch mode.
                //
                if (mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) {
                    partInfo.partitionState = PARTITION_IN_SYNC_PENDING_STATE;
                }
            }

            partInfo.errorOclwrred = false;
            partInfo.gpuInfo = gpuInfoList;
            partInfo.switchInfo = switchInfoList;
            partInfo.trunkConnCount = partHaInfo.trunkconncount();
            mSupportedPartitions.push_back(partInfo);

            FM_LOG_DEBUG("loading partition Id %d and its state %d", partInfo.partitionId, partInfo.partitionState);
        }
    }

    // also rebuild the unsupported partition list
    populateUnsupportedPartitions();

    return true;
}

/*************************************************************************************
 * Interface to query lwrrently supported fabric partitions.
 * 
 * The interface can be used by higher level entities (like GFM) to get the list of 
 * available fabric partitions.
 * 
 **************************************************************************************/
FMIntReturn_t
GlobalFMFabricPartitionMgr::getPartitions(fmFabricPartitionList_t &fmFabricPartitionList)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    FMAutoLock lock(mLock); 
    PartitionInfoList::iterator it;
    int partIdx = 0;
    uint32_t nodeId = 0;

    FM_LOG_INFO("processing request to return list of supported partition list");

    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo tempInfo = (*it);
        fmFabricPartitionList.partitionInfo[partIdx].partitionId = tempInfo.partitionId;
        fmFabricPartitionList.partitionInfo[partIdx].isActive = tempInfo.partitionState;
        fmFabricPartitionList.partitionInfo[partIdx].numGpus = tempInfo.gpuInfo.size();
        // copy all the GPU information
        PartitionGpuInfoList::iterator jit;
        int gpuIdx = 0;
        for (jit = tempInfo.gpuInfo.begin(); jit != tempInfo.gpuInfo.end(); jit++) {
            PartitionGpuInfo tempGpuInfo = (*jit);
            // Note: GPU physical Ids starts from 1 to external APIs. But, internally
            // the IDs starts from 0. So do a +1 to offset the difference
            fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].physicalId = (tempGpuInfo.physicalId) + 1;
            memset(fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid, 0, sizeof(fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid));
            strncpy(fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid,
                    tempGpuInfo.uuid, sizeof(fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid) - 1);
            // fill PCI Bus ID information in LWML format
            FMPciInfo_t *pPciInfo = &tempGpuInfo.dynamicInfo.pciInfo;
            snprintf(fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].pciBusId,
                     FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE, FM_DEVICE_PCI_BUS_ID_FMT,
                     FM_DEVICE_PCI_BUS_ID_FMT_ARGS(pPciInfo));
            fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].numLwLinksAvailable = tempGpuInfo.numLinksAvailable;
            fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].lwlinkLineRateMBps = tempGpuInfo.linkLineRateMBps;
            fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].maxNumLwLinks = 0;

            sharedLWSwitchPartitionInfo *pPartitionCfg = mGfm->mpParser->getSharedLwswitchPartitionCfg(nodeId, tempInfo.partitionId);
            if (pPartitionCfg) {
                // find the max number of lwlinks for this gpu
                uint64_t enabledLinkMask = pPartitionCfg->gpuinfo(gpuIdx).enabledlinkmask();
                while (enabledLinkMask) {
                    fmFabricPartitionList.partitionInfo[partIdx].gpuInfo[gpuIdx].maxNumLwLinks += enabledLinkMask & 1;
                    enabledLinkMask >>= 1;
                }
            }

            gpuIdx++;
        }
        partIdx++;
    }

    fmFabricPartitionList.numPartitions = partIdx;
    fmFabricPartitionList.maxNumPartitions = mGfm->mpParser->getNumConfiguredSharedLwswitchPartition(nodeId);
    return FM_INT_ST_OK;
}

/*************************************************************************************
 * Interface to activate a partition
 * 
 * 
 **************************************************************************************/
FMIntReturn_t
GlobalFMFabricPartitionMgr::activatePartition(uint32 nodeId, fmFabricPartitionId_t partitionId)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    FMAutoLock lock(mLock); 
    FMIntReturn_t fmRetVal;

    FM_LOG_INFO("processing request to activate partition id %d", partitionId);

    if (!isPartitionExists(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
        return FM_INT_ST_BADPARAM;
    }

    if (isPartitionActive(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is already activated", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is already activated", nodeId, partitionId);
        return FM_INT_ST_IN_USE;
    }

    if (isGpuUsedInActivePartitions(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d has GPUs that are already in use", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d has GPUs that are already in use", nodeId, partitionId);
        return FM_INT_ST_IN_USE;
    }

    // clear error flag possibly left from the previous activation/deactivation
    clearPartitionConfigFailure(nodeId, partitionId);

    PartitionInfo partInfo;
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    //
    // request the localFM to disable required GPU LWLinks before initiating any
    // real GPU attach(aka initialization) and link training sequence.
    //
    fmRetVal = mGfm->mpConfig->configSetGpuDisabledLinkMaskForPartition(nodeId, partInfo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to disable desired GPU LWLinks for " NODE_ID_LOG_STR " %d partition id  %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to disable desired GPU LWLinks for " NODE_ID_LOG_STR " %d partition id  %d", nodeId, partitionId);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // request the node's LocalFM to attach all the GPUs.
    fmRetVal = mGfm->mpConfig->configSharedLWSwitchPartitionAttachGPUs(nodeId, partInfo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to attach required GPUs for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to attach required GPUs for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // get attached GPU information from node's LocalFM
    FMGpuInfoMap gpuInfoMap;
    FMExcludedGpuInfoMap excludedGpuInfoMap;
    GFMHelper::getGpuDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, gpuInfoMap, excludedGpuInfoMap);
    // verify whether all the required GPUs in the partition is attached successfully
    if (!validatePartitionGpus(nodeId, partitionId, gpuInfoMap)) {
        FM_LOG_ERROR("required GPUs are not attached or missing for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("required GPUs are not attached or missing for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // get LWLink device information from LWLinkCoereLib driver.
    GlobalFMLWLinkDevRepo lwLinkDevRepo;
    GFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    // do LWLink initialization for the node
    GFMHelper::lwLinkInitializeNode(nodeId, mGfm->mLinkTrainIntf, lwLinkDevRepo);

    // discover all the intra node connections
    GlobalFMLWLinkConnRepo lwLinkConnRepo;
    GFMHelper::lwLinkDiscoverIntraNodeConnOnNode(nodeId, mGfm->mLinkTrainIntf, lwLinkConnRepo);
    
    //
    // as part of connection discovery, LWLink trunk connections will be detected as well.
    // we don't have to train the trunk connections if this partition doesn’t require it.
    // so, remove the same from our connection list before starting the actual training
    //
    filterPartitionLWLinkTrunkConns(nodeId, partitionId, lwLinkConnRepo, lwLinkDevRepo);

    // train all connections
    if ( mGfm->isParallelTrainingEnabled() ) {
         GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(mGfm, mGfm->mLinkTrainIntf, lwLinkConnRepo,
                                                            lwLinkDevRepo, LWLINK_TRAIN_SAFE_TO_HIGH);
    } else {
        GFMHelper::lwLinkTrainIntraNodeConnections(mGfm, mGfm->mLinkTrainIntf, lwLinkConnRepo,
                                                   lwLinkDevRepo, LWLINK_TRAIN_SAFE_TO_HIGH);
    }

    lwLinkConnRepo.dumpAllConnAndStateInfo(mGfm, lwLinkDevRepo);

    // update GPU enum index in the partition for config
    updatePartitionGpuDynamicInfo(nodeId, partitionId, gpuInfoMap);

    // update training failed links information
    fmRetVal = sendActivationTrainingFailedInfoForSwitches(nodeId, partitionId, lwLinkDevRepo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to set LWLink training failed information to LWSwitch driver for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to set LWLink training failed information to LWSwitch driver for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        handlePartitionActivationError(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // verify all the LWLink connections and its state.
    if (!validatePartitionLWLinkConns(nodeId, partitionId, lwLinkConnRepo, gpuInfoMap, lwLinkDevRepo)) {
        FM_LOG_ERROR("LWLink connections are missing or not trained for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("LWLink connections are missing or not trained for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        handlePartitionActivationError(nodeId, partInfo);
        return FM_INT_ST_LWLINK_ERROR;
    }

    // all the pre-conditions are met, and partition information updated.
    // go ahead and configure GPUs and LWSwitches for this partition.
    // We need to update our locally cahed parition information with new
    // gpu enumeration index and bdfs.
    //
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    // enable routing table entries, configure GPU fabric address
    fmRetVal = mGfm->mpConfig->configActivateSharedLWSwitchPartition(nodeId, partInfo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to configure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to configure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        handlePartitionActivationError(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // detach GPUs
    fmRetVal = mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(nodeId, partInfo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to detach GPUs for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to detach GPUs for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        handlePartitionActivationError(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    //
    // All the trunk connections/link will be in SAFE mode if the partition doesn’t require them
    // as we filtered and skipped training. reset those links to completely power them off
    //
    partitionResetFilteredTrunkLWLinks(nodeId, partitionId);

    // request messages are async, the errors could have happened when
    // response messages are received in a different thread.
    // Before return to the API caller, check if any error has oclwrred
    // and handled in the other thread.
    if (isPartitionConfigFailed(nodeId, partitionId)) {
        handlePartitionActivationError(nodeId, partInfo);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // mark the partition state as active
    setPartitionActiveState(nodeId, partitionId, true);

    FM_LOG_INFO(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
    FM_SYSLOG_NOTICE(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
    return FM_INT_ST_OK;
}

/*************************************************************************************
 * Interface to activate a partition with VFs
 * 
 * 
 **************************************************************************************/
FMIntReturn_t
GlobalFMFabricPartitionMgr::activatePartitionWithVfs(uint32 nodeId, fmFabricPartitionId_t partitionId,
                                                     fmPciDevice_t *vfList, unsigned int numVfs)
{
    // Serialize FM LWSwitch shared virtualization related APIs.
    FMAutoLock lock(mLock); 
    FMIntReturn_t fmRetVal;

    FM_LOG_INFO("processing request to activate partition id %d with VFs", partitionId);

    if (!isPartitionExists(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition %d is not supported", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
        return FM_INT_ST_BADPARAM;
    }

    // Check input VfList and numvfs arguments
    if ((vfList == NULL) || (numVfs == 0)) {
        FM_LOG_ERROR(NODE_ID_LOG_STR " %d partition %d activation is called with invalid or empty PCI virtual function BDF information", nodeId, partitionId);
        FM_SYSLOG_ERR(NODE_ID_LOG_STR " %d partition %d activation is called with invalid or empty PCI virtual function BDF information", nodeId, partitionId);
        return FM_INT_ST_BADPARAM;
    }

    if (isPartitionActive(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is already activated", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is already activated", nodeId, partitionId);
        return FM_INT_ST_IN_USE;
    }

    if (isGpuUsedInActivePartitions(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d has GPUs that are already in use", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d has GPUs that are already in use", nodeId, partitionId);
        return FM_INT_ST_IN_USE;
    }

    // Clear error flag possibly left from the previous activation/deactivation
    clearPartitionConfigFailure(nodeId, partitionId);

    PartitionInfo partInfo;
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    // Check the number of VFs being passed
    // return failure if numVfs is less or greater than the number of physical GPUs in the partition
    if (partInfo.gpuInfo.size() != numVfs) {
        FM_LOG_ERROR(NODE_ID_LOG_STR " %d partition %d activation is called with different number of GPU VFs than available in the partition", nodeId, partitionId); 
        FM_SYSLOG_ERR(NODE_ID_LOG_STR " %d partition %d activation is called with different number of GPU VFs than available in the partition", nodeId, partitionId);
        return FM_INT_ST_BADPARAM;
    }

    // Return failure if any GPU in the partition is under reset
    for (PartitionGpuInfoList::iterator it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++)
    {
        PartitionGpuInfo gpuinfo = *it;

        if (isGpuResetInProgress(gpuinfo.uuid)) {
            FM_LOG_ERROR("reset request for GPU uuid %s is in progress", gpuinfo.uuid);
            FM_SYSLOG_ERR("reset request for GPU uuid %s is in progress", gpuinfo.uuid);
            return FM_INT_ST_IN_USE;
        }
    }

    // For a single GPU partition, return success immediately since we don't need
    // to program any switch routing table with FLA, GFID  and other information.
    if (partInfo.gpuInfo.size() == 1) {
        // mark the partition state as active
        setPartitionActiveState(nodeId, partitionId, true);

        FM_LOG_INFO(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
        FM_SYSLOG_NOTICE(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
        return FM_INT_ST_OK;
    }

    // Check for GPU's MIG functionality
    // return failure if GPU's numEnabledLinks is equal to zero.
    for (PartitionGpuInfoList::iterator it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++)
    {
        PartitionGpuInfo gpuinfo = *it;

        if (mGfm->getGpuEnabledLinkMask(gpuinfo.uuid) == 0) {
            FM_LOG_ERROR("No LWLinks are enabled for GPU %s in " NODE_ID_LOG_STR " %d partition id %d", gpuinfo.uuid, nodeId, partitionId);
            FM_SYSLOG_ERR("No LWLinks are enabled for GPU %s in " NODE_ID_LOG_STR " %d partition id %d", gpuinfo.uuid, nodeId, partitionId);
            return FM_INT_ST_IN_USE;
        }
    }

    std::list<GpuGfidInfo> gfids;
    gfids.clear();

    // Get the GFID for all GPUs based on VF's BDF information
    fmRetVal = mGfm->mpConfig->configGetGfidForPartition(nodeId, partInfo, vfList, gfids);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to get gfid value for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to get gfid value for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // Update partition info with gfid and mask values
    setGfidInPartitionGpuDynamicInfo(nodeId, partitionId, gfids);

    // Get updated partition info with GFID/Mask
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    // Enable routing table entries, configure GPU fabric address
    fmRetVal = mGfm->mpConfig->configActivateSharedLWSwitchPartition(nodeId, partInfo);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to configure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to configure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo);
        clearGfidInPartitionGpuDynamicInfo(nodeId, partitionId);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // Notify GPU driver saying that partition is activated.
    fmRetVal = mGfm->mpConfig->configCfgGfidForPartition(nodeId, partInfo, true);
    if (FM_INT_ST_OK != fmRetVal) {
        FM_LOG_ERROR("failed to configure gfid value for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to configure gfid value for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo);
        clearGfidInPartitionGpuDynamicInfo(nodeId, partitionId);
        return FM_INT_ST_GENERIC_ERROR;
    }

    //
    // Request messages are async, the errors could have happened when
    // response messages are received in a different thread.
    // Before return to the API caller, check if any error has oclwrred
    // and handled in the other thread.
    //
    if (isPartitionConfigFailed(nodeId, partitionId)) {
        mGfm->mpConfig->configCfgGfidForPartition(nodeId, partInfo, false);
        mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo);
        clearGfidInPartitionGpuDynamicInfo(nodeId, partitionId);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // Mark the partition state as active
    setPartitionActiveState(nodeId, partitionId, true);

    FM_LOG_INFO(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
    FM_SYSLOG_NOTICE(NODE_ID_LOG_STR " %d partition id %d is activated.", nodeId, partitionId);
    return FM_INT_ST_OK;
}

/*************************************************************************************
 * Interface to deactivate a partition
 * 
 **************************************************************************************/
FMIntReturn_t
GlobalFMFabricPartitionMgr::deactivatePartition(uint32 nodeId, fmFabricPartitionId_t partitionId)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    FMAutoLock lock(mLock); 
    FMIntReturn_t fmResult;

    FM_LOG_INFO("processing request to deactivate " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);

    if (!isPartitionExists(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
        return FM_INT_ST_BADPARAM;
    }

    if (!isPartitionActive(nodeId, partitionId)) {
        FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is not activated ", nodeId, partitionId);
        FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not activated ", nodeId, partitionId);
        return FM_INT_ST_UNINITIALIZED;
    }

    // clear error flag possibly left from the previous activation/deactivation
    clearPartitionConfigFailure(nodeId, partitionId);

    PartitionInfo partInfo;
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);
    fmResult = FM_INT_ST_OK;

    // For vGPU mode, notify GPU driver saying that partition is being deactivated.
    if (mGfm->getFabricMode() == FM_MODE_VGPU) {
        // For a single GPU partition, return success immediately since we don't need
        // to clear any switch routing table entries or need to inform GPU driver.
        if (partInfo.gpuInfo.size() == 1) {
            // mark the partition state as inactive
            setPartitionActiveState(nodeId, partitionId, false);

            FM_LOG_INFO(NODE_ID_LOG_STR " %d partition id %d is deactivated.", nodeId, partitionId);
            FM_SYSLOG_NOTICE(NODE_ID_LOG_STR " %d partition id %d is deactivated.", nodeId, partitionId);

            return FM_INT_ST_OK;
        }

        mGfm->mpConfig->configCfgGfidForPartition(nodeId, partInfo, false);
        clearGfidInPartitionGpuDynamicInfo(nodeId, partitionId);
        // fall through to disable routing next
    }

    // For Shared LWSwitch Mode only
    //
    // reset LWSwitch side lwlinks used in this partition
    // the reset links has to happen before disabling routing to prevent any ACL error
    // as the GPU/GuestVM may be sending traffic or in-flight traffic.
    if ((mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) && (!resetPartitionLWSwitchLinks(nodeId, partitionId))) {
        FM_LOG_ERROR("failed to reset LWSwitch side links for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to reset LWSwitch side links for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        fmResult = FM_INT_ST_LWLINK_ERROR;
        // fall through to disable routing next
    }

    // disable routing table entries even reset lwlinks is not successfull
    if (FM_INT_ST_OK != mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo)) {
        FM_LOG_ERROR("failed to deconfigure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        FM_SYSLOG_ERR("failed to deconfigure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        fmResult = FM_INT_ST_GENERIC_ERROR;
        // fall through to clear error states next
    }

    // mark the partition state as inactive
    setPartitionActiveState(nodeId, partitionId, false);

    FM_LOG_INFO(NODE_ID_LOG_STR " %d partition id %d is deactivated.", nodeId, partitionId);
    FM_SYSLOG_NOTICE(NODE_ID_LOG_STR " %d partition id %d is deactivated.", nodeId, partitionId);

    // request messages are async, the errors could have happened when
    // response messages are received in a different thread.
    // Before return to the API caller, check if any error has oclwrred
    // and handled in the other thread.
    if (isPartitionConfigFailed(nodeId, partitionId) || (fmResult != FM_INT_ST_OK) )
    {
        // Clear pending request before starting next partition deactivation
        mGfm->mpConfig->clearPartitionPendingConfigRequest( nodeId, partitionId );
        clearPartitionConfigFailure(nodeId, partitionId);
        fmResult = FM_INT_ST_GENERIC_ERROR;
    }

    return fmResult;
}

/*************************************************************************************
 * Interface to set a list of already activated partitions after restart
 **************************************************************************************/
FMIntReturn_t
GlobalFMFabricPartitionMgr::setActivatedPartitions(fmActivatedFabricPartitionList_t &fmFabricPartitions)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    FMIntReturn_t fmRetVal;
    FMAutoLock lock(mLock);

    // TODO multi node
    uint32_t nodeId = 0, i, partitionId, pState;
    PartitionInfo partInfo;

    FM_LOG_INFO("processing request to set activated fabric partitions");

    FM_LOG_DEBUG("Set %d number of partitions activate.",
                 fmFabricPartitions.numPartitions);

    for (i = 0; i < fmFabricPartitions.numPartitions; i++) {

        partitionId = fmFabricPartitions.partitionIds[i];

        if ( getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo) == false ) {
            FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
            FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not supported", nodeId, partitionId);
            return FM_INT_ST_BADPARAM;
        }

        pState = getPartitionActiveState(nodeId, partitionId);
        if ((pState == PARTITION_IN_SYNC_PENDING_STATE) || (pState == PARTITION_IN_ACTIVE_STATE)) {
            // mark the partition state as active
            FM_LOG_INFO("marking " NODE_ID_LOG_STR " %d partition id %d as activated after state reloading", nodeId, partitionId);
            setPartitionActiveState(nodeId, partitionId, true);
        } else {
            FM_LOG_ERROR("specified " NODE_ID_LOG_STR " %d partition id %d is not activated", nodeId, partitionId);
            FM_SYSLOG_ERR("specified " NODE_ID_LOG_STR " %d partition id %d is not activated", nodeId, partitionId);
            return FM_INT_ST_BADPARAM;
        }
    }

    // TODO: need to check
    // All the trunk connections/link will be in SAFE mode if the partition doesn’t require them
    // as we filtered and skipped training. reset those links to completely power them off
    //
    // partitionResetFilteredTrunkLWLinks(nodeId, partitionId, false);

    // Disable routing on all not activated partitions that do not use GPUs
    // in activated partitions.
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);

        if (tempInfo.partitionState != PARTITION_IN_SYNC_PENDING_STATE) {
            continue;
        }

        partitionId = tempInfo.partitionId;

        // disabled partition use the same GPUs in the activated partitions
        if (isGpuUsedInActivePartitions(nodeId, partitionId)) {
            // mark the partition state as deactive
            setPartitionActiveState(nodeId, partitionId, false);
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d partition id %d is deactivated", nodeId, partitionId);
            continue;
        }

        // For vGPU mode, notify GPU driver saying that partition is being deactivated.
        if (mGfm->getFabricMode() == FM_MODE_VGPU) {
            //
            // For a single GPU partition, we don't need to clear any
            // switch routing table entries or need to inform GPU driver.
            //
            if (tempInfo.gpuInfo.size() == 1) {
                // mark the partition state as deactive
                setPartitionActiveState(nodeId, partitionId, false);
                FM_LOG_DEBUG(NODE_ID_LOG_STR " %d partition id %d is deactivated", nodeId, partitionId);
                continue;
            }

            mGfm->mpConfig->configCfgGfidForPartition(nodeId, tempInfo, false);
            clearGfidInPartitionGpuDynamicInfo(nodeId, partitionId);
        }

        // the partition is not activated, disable routing entries
        fmRetVal = mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, tempInfo);
        if (FM_INT_ST_OK != fmRetVal) {
            FM_LOG_ERROR("failed to deconfigure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
            FM_SYSLOG_ERR("failed to deconfigure LWSwitch/GPU for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
            return FM_INT_ST_GENERIC_ERROR;
        }

        // request messages are async, the errors could have happened when
        // response messages are received in a different thread.
        // Before return to the API caller, check if any error has oclwrred
        // and handled in the other thread.
        if (isPartitionConfigFailed(nodeId, partitionId)) {
            return FM_INT_ST_GENERIC_ERROR;
        }

        // mark the partition state as deactive
        setPartitionActiveState(nodeId, partitionId, false);

        FM_LOG_DEBUG(NODE_ID_LOG_STR " %d partition id %d is deactivated", nodeId, partitionId);
    }

    mInitDone = true;
    FM_LOG_INFO("completed reloading activated fabric partition list after Shared LWSwitch or vGPU mode restart");
    return FM_INT_ST_OK;
}

bool
GlobalFMFabricPartitionMgr::getSharedLWSwitchPartitionInfo(uint32 nodeId, unsigned int partitionId,
                                                           PartitionInfo &partInfo)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            partInfo = (*it);
            return true;
        }
    }

    // specified partition is not found
    return false;
}

bool
GlobalFMFabricPartitionMgr::mapTopologyPartition(uint32 nodeId, const sharedLWSwitchPartitionInfo &topoPartInfo)
{
    PartitionInfo partInfo;
    PartitionSwitchInfoList switchInfoList;
    PartitionGpuInfoList gpuInfoList;

    switchInfoList.clear();
    gpuInfoList.clear();

    // check whether we already have a partition with specified id
    if (getSharedLWSwitchPartitionInfo(nodeId, topoPartInfo.partitionid(), partInfo)) {
        FM_LOG_ERROR(NODE_ID_LOG_STR " %d partition id %d already exists while creating supported partition list",
                     nodeId, topoPartInfo.partitionid());
        return false;
    }

    // validate whether we detected the required number of GPUs and LWSwitches
    uint32_t requiredGpuCount = topoPartInfo.gpuinfo_size();
    uint32_t requiredSwitchCount =  topoPartInfo.switchinfo_size();

    FMGpuInfoList detectedGpuInfo = mGfm->mGpuInfoMap[nodeId];
    uint32_t detectedGpuCount = detectedGpuInfo.size();

    FMLWSwitchInfoList detectedSwitchInfo = mGfm->mLwswitchInfoMap[nodeId];
    uint32_t detectedSwitchCount = detectedSwitchInfo.size();

    if ((detectedGpuCount < requiredGpuCount) || (detectedSwitchCount < requiredSwitchCount)) {
        // not enough resources to support this partition.
        // this can happen, like 16 GPU partition is not supported on multi-host systems.
        return false;
    }

    // copy each GPU information
    for ( int gpuIdx = 0; gpuIdx < topoPartInfo.gpuinfo_size(); gpuIdx++ ) {
        PartitionGpuInfo gpuInfo;
        const sharedLWSwitchPartitionGpuInfo &topoGpuInfo = topoPartInfo.gpuinfo(gpuIdx);
        gpuInfo.physicalId = topoGpuInfo.physicalid();
        gpuInfo.numEnabledLinks = topoGpuInfo.numenabledlinks();
        gpuInfo.enabledLinkMask = topoGpuInfo.enabledlinkmask();

        if (!mGfm->getGpuUuid(nodeId, gpuInfo.physicalId, gpuInfo.uuid)) {
            return false;
        }

        // fill current variable information like enumeration index and pci BDF
        if (!mGfm->getGpuEnumIndex(nodeId, gpuInfo.physicalId, gpuInfo.dynamicInfo.gpuIndex)) {
            return false;
        }

        if (!mGfm->getGpuPciBdf(nodeId, gpuInfo.dynamicInfo.gpuIndex, gpuInfo.dynamicInfo.pciInfo)) {
            return false;
        }

        // get number of supported links on this GPU
        if (!mGfm->getGpuDiscoveredLinkMask(nodeId, gpuInfo.dynamicInfo.gpuIndex, gpuInfo.discoveredLinkMask)) {
            return false;
        }

        uint32 numActiveLinks;
        if (!mGfm->getGpuNumActiveLWLinks(nodeId, gpuInfo.dynamicInfo.pciInfo, numActiveLinks)) {
            return false;
        }

        //
        // all the LWLinks are enabled on the GPU during FM initialization time. So in ideal case
        // the numActiveLinks should be equal to number of GPU LWLinks. But for DGX-2/HGX-2 systems,
        // certain links has to be disabled for 2, 4 GPU partitions (due to link PLL sharing). This 
        // limitation is not there in new generation switches. So take the lowest of 
        // numEnabledLinks (which is per GPU links to be enabled from topology partition information)
        // and numActiveLinks. In new generation switches, numEnabledLinks and numActiveLinks should be
        // same in ideal case.
        //
 
        gpuInfo.numLinksAvailable = std::min(numActiveLinks, gpuInfo.numEnabledLinks);

        //
        // lwrrently in LWSwitch connected systems, all GPUs has equal number of active LWLinks and
        // all the links has same speed. so getting speed information for one link
        //

        FMLWLinkSpeedInfo linkSpeedInfo;
        if (!mGfm->getGpuLinkSpeedInfo(nodeId, gpuInfo.uuid, linkSpeedInfo)) {
            return false;
        }
        // for single GPU partition (ie if numLinksAvailable = 0), make the speed information to zero
        if (gpuInfo.numLinksAvailable) {
            gpuInfo.linkLineRateMBps = linkSpeedInfo.linkLineRateMBps;
        } else {
            gpuInfo.linkLineRateMBps = 0;
        }

        // set initial gfid and gfidMask as zero
        gpuInfo.dynamicInfo.gfidInfo.gfid = 0;
        gpuInfo.dynamicInfo.gfidInfo.gfidMask = 0;

        gpuInfoList.push_back(gpuInfo);
    }

    // copy each Switch information
    for ( int switchIdx = 0; switchIdx < topoPartInfo.switchinfo_size(); switchIdx++ ) {
        PartitionSwitchInfo switchInfo;
        const sharedLWSwitchPartitionSwitchInfo &topoSwitchInfo = topoPartInfo.switchinfo(switchIdx);
        switchInfo.physicalId = topoSwitchInfo.physicalid();
        switchInfo.archType = mGfm->getSwitchArchType();
        switchInfo.numEnabledLinks = topoSwitchInfo.numenabledlinks();
        switchInfo.enabledLinkMask = topoSwitchInfo.enabledlinkmask();
        switchInfoList.push_back(switchInfo);
    }

    // create the full partition information and save
    partInfo.partitionId = topoPartInfo.partitionid();
    partInfo.partitionState = PARTITION_IN_DEACTIVE_STATE;
    partInfo.errorOclwrred = false;
    partInfo.gpuInfo = gpuInfoList;
    partInfo.switchInfo = switchInfoList;
    partitionMetaDataInfo partMetaData = topoPartInfo.metadata();
    partInfo.trunkConnCount = partMetaData.lwlinkintratrunkconncount();
    mSupportedPartitions.push_back(partInfo);

    return true;
}

bool
GlobalFMFabricPartitionMgr::isPartitionExists(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfo tempPartInfo;
    return getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo);
}

uint32_t
GlobalFMFabricPartitionMgr::getPartitionActiveState(uint32 nodeId, unsigned int partitionId)
{
    uint32_t state = PARTITION_IN_DEACTIVE_STATE;

    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            state = tempInfo.partitionState;
            break;
        }
    }

    return state;
}

void
GlobalFMFabricPartitionMgr::setPartitionActiveState(uint32 nodeId, unsigned int partitionId, bool active)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.partitionState = active ? PARTITION_IN_ACTIVE_STATE : PARTITION_IN_DEACTIVE_STATE;
            break;
        }
    }
}

bool
GlobalFMFabricPartitionMgr::isPartitionActive(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfo tempPartInfo = {0};
    // no need to check return value as we will return false even if the partition is not found
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo);
    return (tempPartInfo.partitionState == PARTITION_IN_ACTIVE_STATE) ? true : false;
}

// put all GPU pysicalIds used in already activated partitions in std::set
void
GlobalFMFabricPartitionMgr::getGpusUsedInActivatePartitions(uint32_t nodeId, std::set<uint32_t> &usedGpus)
{
    usedGpus.clear();

    PartitionInfoList::iterator partIt;
    for (partIt = mSupportedPartitions.begin(); partIt != mSupportedPartitions.end(); ++partIt) {
        PartitionInfo &partInfo = *partIt;
        if (partInfo.partitionState != PARTITION_IN_ACTIVE_STATE)
            continue;

        // add the GPUs in the activated partitions to the set
        PartitionGpuInfoList::iterator gpuIt;
        for (gpuIt = partInfo.gpuInfo.begin(); gpuIt != partInfo.gpuInfo.end(); ++gpuIt)
        {
            PartitionGpuInfo &gpuInfo = *gpuIt;
            usedGpus.insert(gpuInfo.physicalId);
        }
    }
}

//
// check if any GPUs in partitionId are already used in already activated partitions
// return true,  GPUs in partitionId are already used
//        false, GPUs in partitionId are not being used
bool
GlobalFMFabricPartitionMgr::isGpuUsedInActivePartitions(uint32_t nodeId, uint32_t partitionId)
{
    PartitionInfo partInfo = {0};
    if (getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo) == false)
    {
        // partion is not found
        return false;
    }

    std::set<uint32_t> usedGpus;
    getGpusUsedInActivatePartitions(nodeId, usedGpus);

    PartitionGpuInfoList::iterator gpuIt;
    for (gpuIt = partInfo.gpuInfo.begin(); gpuIt != partInfo.gpuInfo.end(); ++gpuIt)
    {
        PartitionGpuInfo &gpuInfo = *gpuIt;
        if (usedGpus.find(gpuInfo.physicalId) != usedGpus.end())
        {
            // find one GPU that is in use
            FM_LOG_DEBUG("find gpu %d is used.", gpuInfo.physicalId);
            return true;
        }
    }

    return false;
}

//
// check whether another partition other than the specified partition ID 
// which uses trunk link is lwrrently active.
// return true: if another partition with trunk link is active.
//        false, no other parition with trunk link is active
bool
GlobalFMFabricPartitionMgr::isAnotherPartitionWithTrunkLinksActive(uint32_t nodeId, uint32_t partitionId)
{
    PartitionInfoList::iterator it;

    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &partInfo = *it;
        if (partInfo.partitionState != PARTITION_IN_ACTIVE_STATE) {
            // not an active partition
            continue;
        }

        if (partInfo.trunkConnCount == 0) {
            // partition is not using any trunk connection
            continue;
        }

        // found an active partition with trunk connection
        // return true if this is not the user specified partition
        if (partInfo.partitionId != partitionId) {
            // another partition with trunk link is active.
            return true;
        }
    }

    // default case, no other partition with trunk link is active lwrrently
    return false;
}

bool
GlobalFMFabricPartitionMgr::validatePartitionGpus(uint32 nodeId, unsigned int partitionId,
                                                 FMGpuInfoMap &gpuInfoMap)
{
    PartitionInfo tempPartInfo;
    FMGpuInfoMap::iterator it = gpuInfoMap.find(nodeId);
    FMGpuInfoList detectedGpuInfoList = it->second;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo)) {
        return false;
    }

    // validate total number of GPUs
    if (tempPartInfo.gpuInfo.size() != detectedGpuInfoList.size()) {
        FM_LOG_ERROR("missing certain GPUs for " NODE_ID_LOG_STR " %d partition id %d, expected GPUs %ld detected GPUs %ld",
                     nodeId, partitionId, tempPartInfo.gpuInfo.size(), detectedGpuInfoList.size());
        return false;
    }

    // validate each GPU by comparing their UUIDs. This will be persistent across
    // each attach/detach cycle by hypervisor.
    PartitionGpuInfoList::iterator jit;
    for (jit = tempPartInfo.gpuInfo.begin(); jit != tempPartInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo partGpuInfo = (*jit);
        bool bFound = false;
        FMGpuInfoList::iterator git;
        for (git = detectedGpuInfoList.begin(); git != detectedGpuInfoList.end(); git++) {
            FMGpuInfo_t FMGpuInfo_t = (*git);
            if (strncasecmp(partGpuInfo.uuid, FMGpuInfo_t.uuid.bytes, sizeof(partGpuInfo.uuid)) == 0) {
                bFound = true;
                break;
            }
        }

        // check whether we found the specified GPU
        if (bFound == false) {
            FM_LOG_ERROR("GPU UUID:%s is missing or not attached for " NODE_ID_LOG_STR " %d partition id %d",
                         partGpuInfo.uuid, nodeId, partitionId);
            return false;
        }
    }

    // all the validation passed.
    return true;
}

uint32_t
GlobalFMFabricPartitionMgr::getGpuNumEnabledLinks(uint32_t nodeId, char *gpuUuid, FMGpuInfoMap &gpuInfoMap)
{
    uint32_t enabledLinkCount = 0;
    if (!gpuUuid) {
        return enabledLinkCount;
    }

    FMGpuInfoMap::iterator gpuMapit = gpuInfoMap.find(nodeId);
    FMGpuInfoList detectedGpuInfoList = gpuMapit->second;

    FMGpuInfoList::iterator git;
    for (git = detectedGpuInfoList.begin(); git != detectedGpuInfoList.end(); git++) {
        FMGpuInfo_t gpuInfo = (*git);
        if (strncmp(gpuUuid, gpuInfo.uuid.bytes, FM_UUID_BUFFER_SIZE) == 0) {
            uint64_t enabledLinkMask = gpuInfo.enabledLinkMask;

            while (enabledLinkMask) {
                enabledLinkCount += enabledLinkMask & 1;
                enabledLinkMask >>= 1;
            }
        }
    }

    return enabledLinkCount;
}

bool
GlobalFMFabricPartitionMgr::validatePartitionLWLinkConns(uint32 nodeId, unsigned int partitionId,
                                                         GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                                                         FMGpuInfoMap &gpuInfoMap,
                                                         GlobalFMLWLinkDevRepo &lwLinkDevRepo)

{
    PartitionInfo tempPartInfo;
    bool retVal = true; // when all the validation looks good
    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo)) {
        return false;
    }

    //
    // check whether all the trunk connections are detected and trained to high speed, if 
    // the partition requires trunk connections
    //
    if (tempPartInfo.trunkConnCount) {
        uint32 detectedTrunkConnCnt = mGfm->mTopoValidator->getIntraNodeTrunkConnCount(nodeId, lwLinkConnRepo);
        if (detectedTrunkConnCnt < tempPartInfo.trunkConnCount) {
            FM_LOG_ERROR(NODE_ID_LOG_STR " %d partition id %d requires %d LWLink trunk connections, detected only %d connections",
                         nodeId, partitionId, tempPartInfo.trunkConnCount, detectedTrunkConnCnt);
            // keep the over-all status as failed, but continue with rest of the validation
            retVal = false;
        }
        if (!mGfm->mTopoValidator->isIntraNodeTrunkConnsActive(nodeId, lwLinkConnRepo, lwLinkDevRepo)) {
            // keep the over-all status as failed, but continue with rest of the validation
            retVal = false;
        }
    }

    // check whether all the access connections are detected and trained to high speed
    uint32 detectedAccessConnCnt = mGfm->mTopoValidator->getAccessConnCount(nodeId, lwLinkConnRepo);

    // compute total access connection count expected
    uint32 accessConnCnt = 0;
    PartitionGpuInfoList::iterator jit;
    for (jit = tempPartInfo.gpuInfo.begin(); jit != tempPartInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo partGpuInfo = (*jit);

        uint32_t gpuInfoNumEnabledLinks = getGpuNumEnabledLinks(nodeId, partGpuInfo.uuid, gpuInfoMap);
        if (gpuInfoNumEnabledLinks != 0) {
            // use the topology numEnabledLinks for validation, if GPU has none zero number of enabled links
            // When GPU is MIG enabled, GPU will have 0 enabled links. The number of topology numEnabledLinks
            // cannot be counted for link verification
            accessConnCnt += partGpuInfo.numEnabledLinks;
        }
    }

    if (detectedAccessConnCnt != accessConnCnt) {
        FM_LOG_ERROR(NODE_ID_LOG_STR " %d partition id %d requires %d LWLink access connections, detected only %d connections",
                     nodeId, partitionId, accessConnCnt, detectedAccessConnCnt);
        // keep the over-all status as failed, but continue with rest of the validation
        retVal = false;
    }
    if (!mGfm->mTopoValidator->isAccessConnsActive(nodeId, lwLinkConnRepo, lwLinkDevRepo)) {
        // keep the over-all status as failed, but continue with rest of the validation
        retVal = false;
    }

    // return the final validation status
    return retVal;
}

void
GlobalFMFabricPartitionMgr::updatePartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId,
                                                         FMGpuInfoMap &gpuInfoMap)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            // found the required partition
            break;
        }
    }

    if (it == mSupportedPartitions.end()) {
        // not found the partition, this shouldn't happen
        return;
    }

    PartitionInfo &partInfo = (*it);

    FMGpuInfoMap::iterator gpuMapit = gpuInfoMap.find(nodeId);
    FMGpuInfoList detectedGpuInfoList = gpuMapit->second;
    // now update each GPU's dynamic information by comparing their UUIDs
    PartitionGpuInfoList::iterator jit;
    for (jit = partInfo.gpuInfo.begin(); jit != partInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo &partGpuInfo = (*jit);
        // find the corresponding GPU from the lwrrently attached GPUs.
        FMGpuInfoList::iterator git;
        for (git = detectedGpuInfoList.begin(); git != detectedGpuInfoList.end(); git++) {
            FMGpuInfo_t FMGpuInfo_t = (*git);
            if (strncasecmp(partGpuInfo.uuid, FMGpuInfo_t.uuid.bytes, sizeof(partGpuInfo.uuid)) == 0) {
                partGpuInfo.dynamicInfo.gpuIndex = FMGpuInfo_t.gpuIndex;
                partGpuInfo.dynamicInfo.pciInfo = FMGpuInfo_t.pciInfo;
                break;
            }
        }
   }
}

void
GlobalFMFabricPartitionMgr::setGfidInPartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId,
                                                             std::list<GpuGfidInfo> &gfidList)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            // found the required partition
            break;
        }
    }

    if (it == mSupportedPartitions.end()) {
        // not found the partition, this shouldn't happen
        return;
    }

    PartitionInfo &partInfo = (*it);

    if (gfidList.size() != partInfo.gpuInfo.size() ) {
        FM_LOG_ERROR("number of gfids are less or more than the number of GPUs in the " NODE_ID_LOG_STR " %d partition %d", nodeId, partitionId);
        return;
    }

    // now update each GPU's dynamic information with gfid and mask values.
    PartitionGpuInfoList::iterator git;
    std::list<GpuGfidInfo>::iterator gfid = gfidList.begin();

    // The order of GPUs in GpuGfidInfo should be same as PartitionGpuInfo
    for (git = partInfo.gpuInfo.begin(); git != partInfo.gpuInfo.end(); git++, gfid++) {
        PartitionGpuInfo &partGpuInfo = (*git);
        GpuGfidInfo &gfidInfo = (*gfid);

        partGpuInfo.dynamicInfo.gfidInfo.gfid = gfidInfo.gfid;
        partGpuInfo.dynamicInfo.gfidInfo.gfidMask = gfidInfo.gfidMask;
   }
}

void
GlobalFMFabricPartitionMgr::clearGfidInPartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            // found the required partition
            break;
        }
    }

    if (it == mSupportedPartitions.end()) {
        // not found the partition, this shouldn't happen
        return;
    }

    PartitionInfo &partInfo = (*it);

    // now update each GPU's dynamic information with gfid and mask values.
    PartitionGpuInfoList::iterator git;

    for (git = partInfo.gpuInfo.begin(); git != partInfo.gpuInfo.end(); git++) {
        PartitionGpuInfo &partGpuInfo = (*git);

        partGpuInfo.dynamicInfo.gfidInfo.gfid = 0;
        partGpuInfo.dynamicInfo.gfidInfo.gfidMask = 0;
   }
}

bool
GlobalFMFabricPartitionMgr::resetPartitionLWSwitchLinks(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfo partInfo;
    int retVal = true;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return false;
    }

    //
    // before starting reset, switch off all the active trunk connections (ie 16 GPU case)
    // otherwise while resetting one endpoint(link), other connected endpoint(link) will
    // generate lwlink fatal error. Access connections are fine as we don't have the
    // GPU with us any way and GPU will get a full SBR by hypervisor.
    //
    partitionPowerOffTrunkLWLinkConns(nodeId, partitionId);

    // reset corresponding LWSwitch links.
    PartitionSwitchInfoList::iterator it;
    for (it = partInfo.switchInfo.begin(); it != partInfo.switchInfo.end(); it++ ) {
        PartitionSwitchInfo switchInfo = (*it);
        uint64 resetLinkMask = 0;

        //
        // reset should be in pairs for Willow. so compute the mask considering the odd/even pair
        //
        if (switchInfo.archType == LWSWITCH_ARCH_TYPE_SV10) {
            uint64 tempEnabledMask = switchInfo.enabledLinkMask;
            for (uint64_t linkId = 0; tempEnabledMask != 0; linkId +=2, tempEnabledMask >>= 2) {
                if ((tempEnabledMask & 0x3) != 0) {
                    // rebuild the mask
                    resetLinkMask |= (BIT64(linkId) | BIT64(linkId + 1));
                }
            }
        } else {
            //
            // in LimeRock, we need to reset all the links which are enabled for the partition
            // so reset mask will be enabled link mask
            //
            resetLinkMask = switchInfo.enabledLinkMask;
        }
        //
        // the above computed mask will have trunk link as well (for partitions which requires trunk)
        // however, leave the trunk if another partition using trunk is active (like across base board VMs)
        // find the trunk mask and clear it if so
        // 
        if (isAnotherPartitionWithTrunkLinksActive(nodeId, partitionId)) {
            uint64 trunkLinkMask;
            mGfm->mpParser->getSwitchTrunkLinkMask(nodeId, switchInfo.physicalId, trunkLinkMask);
            resetLinkMask = (resetLinkMask & ~trunkLinkMask);
            //
            // In DGX-2/HGX-2 some switches has only trunk links enabled (for across Base board connectivity)
            // if another partition using trunk link is active, then no other links to reset on
            //those LWSwitches, so skip the rest 
            //
            if (resetLinkMask == 0) {
                // no links to reset, skip the command to localFM
                continue;
            }
        }

        // go ahead and reset the links used for this partition
        int tempRet = GFMHelper::lwLinkSendResetSwitchLinks(nodeId, switchInfo.physicalId,
                                                            resetLinkMask, mGfm->mLinkTrainIntf);
        if (tempRet) {
            FM_LOG_ERROR("failed to do LWLink reset for LWSwitch " NODE_ID_LOG_STR " %d physical id %d ", nodeId, switchInfo.physicalId );
            // indicate the overall status, but continue with other switches.
            retVal = false;
        }
    }

    return retVal;
}

void
GlobalFMFabricPartitionMgr::filterPartitionLWLinkTrunkConns(uint32 nodeId, unsigned int partitionId,
                                                           GlobalFMLWLinkConnRepo &linkConnRepo,
                                                           GlobalFMLWLinkDevRepo &linkDevRepo)
{
    PartitionInfo partInfo;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount != 0) {
        // no trunk connection to remove, ie 16 GPU VM or VMs spanning across two base boards.
        return;
    }

    // don't touch trunk connection if another partition using trunk is active (like across base board VMs)
    if (isAnotherPartitionWithTrunkLinksActive(nodeId, partitionId)) {
         return;
    }

    // first get the list of intra connections for this node.
    // then remove the trunk connections.
    LWLinkIntraConnMap &tempIntraConnMap = linkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = tempIntraConnMap.find(nodeId);
    if (it == tempIntraConnMap.end()) {
        // no entry for the specified node
        return;
    }
    FMLWLinkDetailedConnInfoList tempIntraConnList = it->second;
    FMLWLinkDetailedConnInfoList::iterator connIt;

    // iterate over all the connections in the node and remove trunk connections
    for (connIt = tempIntraConnList.begin(); connIt != tempIntraConnList.end();) {
        FMLWLinkDetailedConnInfo *lwLinkConn = *connIt;
        if (partitionIsLWLinkTrunkConnection(lwLinkConn, linkDevRepo)) {
            // remove this connection
            connIt = tempIntraConnList.erase( connIt );
            // free the memory
            delete lwLinkConn;
        } else {
            // move to next connection
            ++connIt;
        }
    }

    // now all the trunk connections are removed from tempIntraConnList for the specified
    // node. remove the same from linkConnRepo and re-add the new list.

    // tempIntraConnMap is a reference.
    tempIntraConnMap.erase(it);
    tempIntraConnMap.insert(std::make_pair(nodeId, tempIntraConnList));

}

bool
GlobalFMFabricPartitionMgr::partitionIsLWLinkTrunkConnection(FMLWLinkDetailedConnInfo *lwLinkConn,
                                                            GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    FMLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    FMLWLinkDevInfo end0LWLinkDevInfo; 
    FMLWLinkDevInfo end1LWLinkDevInfo;

    // get device details as seen by LWLinkCoreLib driver.
    linkDevRepo.getDeviceInfo(endPoint0.nodeId, endPoint0.gpuOrSwitchId, end0LWLinkDevInfo);
    linkDevRepo.getDeviceInfo(endPoint1.nodeId, endPoint1.gpuOrSwitchId, end1LWLinkDevInfo);

    if (end0LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch &&
        end1LWLinkDevInfo.getDeviceType() == lwlink_device_type_lwswitch) {
        return true;
    }

    // not a trunk connection
    return false;
}

void
GlobalFMFabricPartitionMgr::partitionResetFilteredTrunkLWLinks(uint32 nodeId,
                                                              unsigned int partitionId)
{
    PartitionInfo partInfo;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount != 0) {
        // all the trunk connections are required and nothing to reset
        // ie 16 GPU VM or VMs span across two base board.
        return;
    }

    // don't touch trunk connection if another partition using trunk is active (like across base board VMs)
    if (isAnotherPartitionWithTrunkLinksActive(nodeId, partitionId)) {
        return;
    }

    //
    // all the trunk links are in safe mode now and we need to reset across all the
    // switches. so taking the switches from partition context is not enough
    // so go through all the detected switches for the node.
    //
    // Note/TODO: Make this platform agnostic/configurable.
    //
    FMLWSwitchInfoMap::iterator it = mGfm->mLwswitchInfoMap.find(nodeId);
    if (it == mGfm->mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return;
    }

    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfo switchInfo = switchList.front();

    // get the trunk port mask derived from topology file
    uint64 trunkLinkMask;
    mGfm->mpParser->getSwitchTrunkLinkMask(nodeId, switchInfo.physicalId, trunkLinkMask);
    if (!(switchInfo.enabledLinkMask & trunkLinkMask)) {
        // trunk links are already disabled as part of multi-host. no need to do reset
        return;
    }

    // go ahead and reset each trunk links.
    FMLWSwitchInfoList::iterator jit;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        FMLWSwitchInfo switchInfo = (*jit);
        uint64 trunkLinkMask;
        mGfm->mpParser->getSwitchTrunkLinkMask(nodeId, switchInfo.physicalId, trunkLinkMask);
        // go ahead and reset the trunk links
        int tempRet = GFMHelper::lwLinkSendResetSwitchLinks(nodeId, switchInfo.physicalId,
                                                            trunkLinkMask, mGfm->mLinkTrainIntf);
        if (tempRet) {
            FM_LOG_ERROR("failed to do LWLink trunk link reset for " NODE_ID_LOG_STR " %d LWSwitch physical id %d ", nodeId, switchInfo.physicalId );
            // best effort, just log the error and continue.
        }
    }
}

void
GlobalFMFabricPartitionMgr::partitionPowerOffTrunkLWLinkConns(uint32 nodeId,
                                                             unsigned int partitionId)
{
    PartitionInfo partInfo;
    int retVal = true;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount == 0) {
        // no trunk connections used for this partition,
        // ie not a 16 GPU VM or VMs span across two base board
        return;
    }

    // don't touch trunk connection if another partition using trunk is active (like across base board VMs)
    if (isAnotherPartitionWithTrunkLinksActive(nodeId, partitionId)) {
        return;
    }

    //
    // instead of caching the trunk connections, we are querying it again from the driver.
    //

    // get LWLink device information from LWLinkCoereLib driver.
    GlobalFMLWLinkDevRepo lwLinkDevRepo;
    GFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    // get the list of all the intra node connections
    GlobalFMLWLinkConnRepo lwLinkConnRepo;
    GFMHelper::lwLinkGetIntraNodeConnOnNodes(nodeId, mGfm->mLinkTrainIntf, lwLinkConnRepo);

    // reset all the trunk connections.
    // Note: No need to filter the detected connections as we should only get trunk connections
    // as the GPU are detached from service VM and associated access connections are removed.
    GFMHelper::lwLinkTrainIntraNodeConnections(mGfm, mGfm->mLinkTrainIntf, lwLinkConnRepo,
                                               lwLinkDevRepo, LWLINK_TRAIN_TO_OFF);
}

void
GlobalFMFabricPartitionMgr::clearPartitionConfigFailure(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.errorOclwrred = false;
            FM_LOG_DEBUG("clearPartitionConfigFailure " NODE_ID_LOG_STR " %d partition %d", nodeId, partitionId);
            break;
        }
    }
}

void
GlobalFMFabricPartitionMgr::setPartitionConfigFailure(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.errorOclwrred = true;
            FM_LOG_DEBUG("setPartitionConfigFailure " NODE_ID_LOG_STR " %d partition %d", nodeId, partitionId);
            break;
        }
    }
}

bool
GlobalFMFabricPartitionMgr::isPartitionConfigFailed(uint32 nodeId, uint32_t partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            return tempInfo.errorOclwrred;
        }
    }

    return false;
}

// Get the first active partition id for the given LWSwitch and PortNumber pair
// if found, a valid partitionId is returned
// if not found, ILWALID_FABRIC_PARTITION_ID is returned
uint32_t
GlobalFMFabricPartitionMgr::getActivePartitionIdForLWSwitchPort(uint32_t nodeId,
                                                               uint32_t physicalId,
                                                               uint32_t portNum)
{
    PartitionInfoList::iterator partIt;
    for (partIt = mSupportedPartitions.begin();
         partIt != mSupportedPartitions.end();
         ++partIt) {

        PartitionInfo &partInfo = *partIt;
        if (partInfo.partitionState != PARTITION_IN_ACTIVE_STATE) continue;

        PartitionSwitchInfoList::iterator switchIt;
        for (switchIt = partInfo.switchInfo.begin();
             switchIt != partInfo.switchInfo.end();
             ++switchIt) {

            PartitionSwitchInfo &switchInfo = *switchIt;
            if (switchInfo.physicalId != physicalId) continue;

            if ((switchInfo.enabledLinkMask & ((uint64_t)1 << portNum)) != 0) {
                // found the first matching link
                // the same LWSwitch and PortNumber pair cannot belong to two
                // active partitions at the same time
                return partInfo.partitionId;
            }
        }
    }

    return ILWALID_FABRIC_PARTITION_ID;
}

void
GlobalFMFabricPartitionMgr::handlePartitionActivationError(uint32_t nodeId, PartitionInfo &partInfo)
{
    // Disable routing
    mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo);

    // reset access lwlinks
    resetPartitionLWSwitchLinks(nodeId, partInfo.partitionId);

    // reset trunk lwlinks
    partitionResetFilteredTrunkLWLinks(nodeId, partInfo.partitionId);

    // Detach GPUs
    mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(nodeId, partInfo);

    // Clear any pending request
    mGfm->mpConfig->clearPartitionPendingConfigRequest(nodeId, partInfo.partitionId);

    // clear the error on the partition
    clearPartitionConfigFailure(nodeId, partInfo.partitionId);
}

// if no LWSwitch that should connect to the specified GPU is detected
// return true, consider that this GPU is on a non detected baseboard.
bool
GlobalFMFabricPartitionMgr::isGpuOnNotDetectedBasebard(uint32_t nodeId, uint32_t physicalId)
{
    GpuKeyType gpuKey;
    gpuKey.nodeId = nodeId;
    gpuKey.physicalId = physicalId;

    // get all LWSwitch connected to this GPU
    std::map <SwitchKeyType, uint64_t> connectedSwitches;
    std::map <SwitchKeyType, uint64_t>::iterator it;
    mGfm->mpParser->getSwitchPortMaskToGpu(gpuKey, connectedSwitches);

    for (it = connectedSwitches.begin(); it != connectedSwitches.end(); it++) {
        SwitchKeyType switchKey = it->first;
        FMPciInfo_t pciInfo;

        if (mGfm->getLWSwitchPciBdf(gpuKey.nodeId, switchKey.physicalId, pciInfo) == true) {
            // a switch connected to this GPU is detected
            return false;
        }
    }

    // no LWSwitch that should connect to this GPU is detected
    return true;
}

/*
 * There is no need to save in HA state, as all information can be derived from the known
 * information after restart, that is
 * - supported partitions (loaded from the HA state)
 * - GFM switchInfoMap and excludedSwitchInfoMap, both re-fetched from LFM after restart
 * - all partitions from the topology file
 */
void
GlobalFMFabricPartitionMgr::populateUnsupportedPartitions()
{
    // TODO multi node
    uint32_t nodeId = 0;

    FMLWSwitchInfoMap switchInfoMap = mGfm->getLwSwitchInfoMap();
    FMExcludedLWSwitchInfoMap excludedSwitchInfoMap = mGfm->getExcludedLwswitchInfoMap();
    bool oneBaseBoard = mGfm->isSingleBaseboard(nodeId);

    uint32_t partIdx = 0;
    std::map <PartitionKeyType, sharedLWSwitchPartitionInfo *>::iterator it;

    // iterator all configured partitions
    for (it = mGfm->mpParser->sharedLwswitchPartitionCfg.begin();
         it != mGfm->mpParser->sharedLwswitchPartitionCfg.end();
         it++) {

        sharedLWSwitchPartitionInfo *partitionCfg = it->second;
        uint32_t partitionId = partitionCfg->partitionid();

        PartitionInfo partInfo;
        if ( getSharedLWSwitchPartitionInfo(0, partitionId, partInfo) == true ) {
            // this configured partition is supported
            continue;
        }

        if (oneBaseBoard) {

            // check cross base board partitions
            if (partitionCfg->has_metadata() &&
                partitionCfg->metadata().lwlinkintratrunkconncount() > 0) {
                // Any configured partition that uses trunk ports are not available
                // on a one baseboard or half system
                // Do not count these as unsupported partitions
                continue;
            }

            bool allGpuOnNotDetectedBasebard = true;
            for (int i = 0; i < partitionCfg->gpuinfo_size(); i++) {
                sharedLWSwitchPartitionGpuInfo gpuInfo = partitionCfg->gpuinfo(i);
                if (isGpuOnNotDetectedBasebard(nodeId, gpuInfo.physicalid()) == false) {
                    // the GPU is not on non existing base board
                    allGpuOnNotDetectedBasebard = false;
                    break;
                }
            }

            if (allGpuOnNotDetectedBasebard == true) {
                // all GPUs in this partition are on not detected base board
                // Do not count this as unsupported partitions
                continue;
            }
        }

        // Add the partition to unsupported partition list
        fmUnsupportedFabricPartitionInfo_t &partition = mUnsupportedPartitionList.partitionInfo[partIdx];
        partition.partitionId = partitionId;

        // get all the GPUs in the unsupported partition
        for (int32_t  i = 0; i < partitionCfg->gpuinfo_size(); i++) {
            sharedLWSwitchPartitionGpuInfo gpuInfo = partitionCfg->gpuinfo(i);

            // partition APIs use 1 based GPU physicalId
            partition.gpuPhysicalIds[i] = gpuInfo.physicalid() + 1;
        }
        partition.numGpus = partitionCfg->gpuinfo_size();

        // increase the unsupported partition idx
        partIdx++;
    }

    mUnsupportedPartitionList.numPartitions = partIdx;
}

FMIntReturn_t
GlobalFMFabricPartitionMgr::getUnsupportedPartitions(fmUnsupportedFabricPartitionList_t &fmFabricPartitionList)
{
    fmFabricPartitionList.numPartitions = mUnsupportedPartitionList.numPartitions;

    memcpy(fmFabricPartitionList.partitionInfo,
            mUnsupportedPartitionList.partitionInfo,
            sizeof(fmUnsupportedFabricPartitionInfo_t)*FM_MAX_FABRIC_PARTITIONS);

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFMFabricPartitionMgr::sendActivationTrainingFailedInfoForSwitches(uint32 nodeId,
                                                                        unsigned int partitionId,
                                                                        GlobalFMLWLinkDevRepo &lwLinkDevRepo)
{
    PartitionInfo partInfo;
    FMIntReturn_t retVal = FM_INT_ST_OK;

    //
    // Note: out-of-band is not supported in Willow, so treat as success
    //
    if (LWSWITCH_ARCH_TYPE_SV10 == mGfm->getSwitchArchType()) {
        return FM_INT_ST_OK;
    }

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return FM_INT_ST_GENERIC_ERROR;
    }

    PartitionSwitchInfoList::iterator it;
    for (it = partInfo.switchInfo.begin(); it != partInfo.switchInfo.end(); it++) {
        PartitionSwitchInfo switchInfo = (*it);
        FMPciInfo_t switchPciInfo;
        //
        // first get the corresponding LWLink device information.
        //
        // Note: LWSwitches are not removed from Service VM after FM initialization, so looking up through the
        // cached information in GlobalFM.
        //

        if (!mGfm->getLWSwitchPciBdf(nodeId, switchInfo.physicalId, switchPciInfo)) {
            // this is not expected, treat as error
            return FM_INT_ST_GENERIC_ERROR;
        }

        FMLWLinkDevInfo switchLwLinkDevInfo;
        if (lwLinkDevRepo.getDeviceInfo(nodeId, switchPciInfo, switchLwLinkDevInfo) == false) {
            // this is not expected, treat as error
            std::stringstream outStr;
            outStr << "unable to find LWSwitch pci bus id:" << switchPciInfo.busId << " physical id:"
                   << switchInfo.physicalId << " information in LWLink driver context";
            FM_LOG_ERROR("%s", outStr.str().c_str());
            FM_SYSLOG_ERR("%s", outStr.str().c_str());
            return FM_INT_ST_GENERIC_ERROR;
        }

        retVal = sendActivationTrainingFailedInfoForSwitch(nodeId, switchInfo, switchLwLinkDevInfo);
        if (FM_INT_ST_OK != retVal) {
            // error already logged.
            return retVal;
        }
    }

    // successfully updated all the switches link failed information during partition activation
    return retVal;
}

FMIntReturn_t
GlobalFMFabricPartitionMgr::sendActivationTrainingFailedInfoForSwitch(uint32_t nodeId, 
                                                                       PartitionSwitchInfo &partSwitchInfo,
                                                                       FMLWLinkDevInfo &switchLwLinkDevInfo)
{
    uint64 trainingFailedLinkMask0 = 0;
    uint64 trainingAttemptedLinkMask0 = partSwitchInfo.enabledLinkMask;
    uint64 tempEnabledMask = partSwitchInfo.enabledLinkMask;
    for (uint32 linkIdx = 0; linkIdx < LWLINK_MAX_DEVICE_CONN; linkIdx++) {
        // skip if the link is not enabled or used for this partition
        if (!(tempEnabledMask & ((uint64)1 << linkIdx))) {
            continue;
        }

        // this link is required for the partition, check whether it is trained to active
        if (false == switchLwLinkDevInfo.isLinkActive(linkIdx)) {
            // this link was supposed to be active, report it as failed
            trainingFailedLinkMask0 |= BIT64(linkIdx);
        }
    }

    // send the computed link mask to corresponding localFM
    int tempRet = GFMHelper::lwLinkSendSwitchTrainingFailedLinkInfo(nodeId, partSwitchInfo.physicalId,
                                                                    trainingAttemptedLinkMask0, trainingFailedLinkMask0,
                                                                    mGfm->mLinkTrainIntf);
    if (tempRet) {
        FM_LOG_ERROR("failed to set LWSwitch link retraining failed information for " NODE_ID_LOG_STR " %d LWSwitch physical id %d ", nodeId, partSwitchInfo.physicalId );
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

// Check if a given GPU is already used in any active partition.
// return true,  GPU is already used in active partition.
//        false, GPU is not being used in any active partition.
bool
GlobalFMFabricPartitionMgr::isGpuInActivePartition(uint32_t nodeId, char *gpuUuid)
{
    uint32_t physicalId;
    std::set<uint32_t> usedGpus;

    if ( !gpuUuid || ( mGfm->getGpuPhysicalId(nodeId, gpuUuid, physicalId) == false ) )
    {
        FM_LOG_ERROR("Invalid GPU UUID or Physical ID");
        return false;
    }

    getGpusUsedInActivatePartitions(nodeId, usedGpus);

    if (usedGpus.find(physicalId) != usedGpus.end())
    {
        // Given GPU is part of an active partition
        FM_LOG_DEBUG("GPU with " NODE_ID_LOG_STR " %d and physical id %d is already in use", nodeId, physicalId);
        return true;
    }

    return false;
}

void
GlobalFMFabricPartitionMgr::addGpuToResetPendingList(char *gpuUuid)
{
    FMAutoLock lock(mLock); 
    FMUuid_t uuid = {{0}};
    strncpy(uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    FM_LOG_DEBUG("add GPU %s to mResetPendingGpus list", gpuUuid);
    mResetPendingGpus.insert(uuid);
}

void
GlobalFMFabricPartitionMgr::remGpuFromResetPendingList(char *gpuUuid)
{
    FMAutoLock lock(mLock); 
    FMUuid_t uuid = {{0}};
    strncpy(uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    FM_LOG_DEBUG("remove GPU %s from mResetPendingGpus list", gpuUuid);
    mResetPendingGpus.erase(uuid);
}

bool
GlobalFMFabricPartitionMgr::isGpuResetInProgress(char *gpuUuid)
{
    FMUuid_t uuid = {{0}};
    strncpy(uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    if (mResetPendingGpus.find(uuid) != mResetPendingGpus.end()) {
        FM_LOG_DEBUG("GPU %s is in reset", gpuUuid);
        return true;
    }

    FM_LOG_DEBUG("GPU %s is not in reset", gpuUuid);
    return false;
}

// check if a switch is used in a partition
bool
GlobalFMFabricPartitionMgr::getSwitchEnabledLinkMaskForPartition(uint32_t partitionId, SwitchKeyType switchKey,
                                                                 uint64_t &enabledLinkMask)
{
    enabledLinkMask = 0;

    PartitionInfo partInfo;
    if (getSharedLWSwitchPartitionInfo(switchKey.nodeId, partitionId, partInfo) == false) {
        // partition does not exist
        return false;
    }

    PartitionSwitchInfoList::iterator it;
    PartitionSwitchInfoList &switchList = partInfo.switchInfo;

    for (it = switchList.begin(); it != switchList.end(); it++) {
        PartitionSwitchInfo &switchInfo = *it;
        if (switchInfo.physicalId == switchKey.physicalId) {
            enabledLinkMask = switchInfo.enabledLinkMask;
            return true;
        }
    }

    return false;
}

// check if a GPU is used in a partition
bool
GlobalFMFabricPartitionMgr::isGpuUsedInPartition(uint32_t partitionId, GpuKeyType gpuKey)
{
    PartitionInfo partInfo;
    if (getSharedLWSwitchPartitionInfo(gpuKey.nodeId, partitionId, partInfo) == false) {
        // partition does not exist
        return false;
    }

    PartitionGpuInfoList::iterator it;
    PartitionGpuInfoList &gpuList = partInfo.gpuInfo;

    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        PartitionGpuInfo &gpuInfo = *it;
        if (gpuInfo.physicalId == gpuKey.physicalId) {
            return true;
        }
    }

    return false;
}
