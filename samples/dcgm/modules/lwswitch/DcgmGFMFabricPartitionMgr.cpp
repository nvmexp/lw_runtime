#include <iostream>
#include <sstream>

#include "logging.h"
#include "DcgmFMCommon.h"
#include "DcgmGFMHelper.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFabricParser.h"
#include "DcgmGFMFabricPartitionMgr.h"
#include "DcgmLogging.h"
#include "DcgmFMAutoLock.h"

DcgmGFMFabricPartitionMgr::DcgmGFMFabricPartitionMgr(DcgmGlobalFabricManager *pGfm)
{
    mGfm = pGfm;
    mSupportedPartitions.clear();
    lwosInitializeCriticalSection(&mLock);

    mpHaState = new fabricmanagerHA::sharedFabricPartiontionInfo;
    mInitDone = false;
}

DcgmGFMFabricPartitionMgr::~DcgmGFMFabricPartitionMgr()
{
    delete mpHaState;
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
DcgmGFMFabricPartitionMgr::buildPartitionMappings()
{
    if (mGfm->isRestart() == true) {
        PRINT_ERROR("", "This should not be called at restart.");
        return false;
    }

    // lwrrently look for only node 0.
    // Note: re-visit during multi-node support
    std::map <NodeKeyType, NodeConfig *>::iterator it = mGfm->mpParser->NodeCfg.begin();
    if(it == mGfm->mpParser->NodeCfg.end()) {
        PRINT_WARNING("", "fabric node 0 not found while building fabric partition mapping.");
        return false;
    }

    // iterate through all the partitions specified in the topology file
    NodeConfig *pNodeCfg = it->second;
    for (int idx = 0; idx < pNodeCfg->partitionInfo.sharedlwswitchinfo_size(); idx++) {
        const sharedLWSwitchPartitionInfo &topoPartInfo = pNodeCfg->partitionInfo.sharedlwswitchinfo(idx);
        if (!mapTopologyPartition(pNodeCfg->nodeId, topoPartInfo)) {
            PRINT_WARNING("%d", "failed to create fabric partition mapping for partition id %d.",
                           topoPartInfo.partitionid());
            continue;
        }
    }

    mInitDone = true;
    return true;
}

/*
 * get partition mapping to mpHaState protobuf
 */
bool
DcgmGFMFabricPartitionMgr::getSharedFabricHaState(fabricmanagerHA::sharedFabricPartiontionInfo &haState)
{
    if (mSupportedPartitions.size() == 0) {
        // mSupportedPartitions is not ready yet
        PRINT_ERROR("", "Shared fabric partition info is not ready yet.");
        return false;
    }

    fabricmanagerHA::infoHeader *haHeader = new fabricmanagerHA::infoHeader;
    mpHaState->set_allocated_header(haHeader);
    fabricmanagerHA::nodePartitionInfo *haNodeInfo = mpHaState->add_nodelist();

    // set to the lwrrently supported version
    haHeader->set_version(SHARED_FABRIC_PARTITION_HA_STATE_VER);

    // TODO, Need to revisit for multi-node
    haNodeInfo->set_nodeid(0);

    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &partInfo = *it;

        fabricmanagerHA::partitionInfo *haPartInfo = haNodeInfo->add_partitionlist();
        haPartInfo->set_partitionid(partInfo.partitionId);
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
       }

       // save all LWSwitches in the partition
       PartitionSwitchInfoList::iterator switchIt;
       for (switchIt = partInfo.switchInfo.begin();
            switchIt != partInfo.switchInfo.end();
            switchIt++) {
           PartitionSwitchInfo &partSwitchInfo = (*switchIt);

           fabricmanagerHA::partitionSwitchInfo *switchHaInfo = haPartInfo->add_switchlist();
           switchHaInfo->set_physicalid(partSwitchInfo.physicalId);
           switchHaInfo->set_numenabledlinks(partSwitchInfo.numEnabledLinks);
           switchHaInfo->set_enabledlinkmask(partSwitchInfo.enabledLinkMask);
       }
    }

    haState = *mpHaState;
    return true;
}

/*
 * Validate a previously saved partition mapping
 */
bool
DcgmGFMFabricPartitionMgr::validateSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState)
{
    // version check
    if (savedHaState.header().version() != SHARED_FABRIC_PARTITION_HA_STATE_VER) {
        // TODO: process message upgrade/down grade
    }

    // number of node check
    if (savedHaState.nodelist_size() != (int)mGfm->mpParser->NodeCfg.size()) {
        PRINT_ERROR("%d %d", "Saved node size %d is different from topology node size %d",
                    (int)savedHaState.nodelist_size(), (int)mGfm->mpParser->NodeCfg.size());
        // comment out the syslog on multi node before multi node is supported
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
        PRINT_ERROR("", "Fabric node 0 not found in the topology.");
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
        PRINT_ERROR("", "Fabric node 0 not found in the save HA state.");
        return false;
    }

    // check if the number of saved partitions are more than the number
    // defined for the node
    if (pNodeCfg->partitionInfo.sharedlwswitchinfo_size() < pNodeHaInfo->partitionlist_size()) {
        PRINT_ERROR("%d %d", "Saved partition size %d is more than topology partition size %d",
                    (int)pNodeHaInfo->partitionlist_size(),
                    (int)pNodeCfg->partitionInfo.sharedlwswitchinfo_size());
        FM_SYSLOG_ERR("The Partition size %d saved in the state file is more than"
                      "the partition size %d defined in the topology file.",
                     (int)pNodeHaInfo->partitionlist_size(),
                     (int)pNodeCfg->partitionInfo.sharedlwswitchinfo_size());
        return false;
    }

    return true;
}

/*
 * From a saved partition states, restore shared partition mapping states.
 * The saved partition states need to be validated before calling this.
 */
bool
DcgmGFMFabricPartitionMgr::loadSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState)
{
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
                strncpy(gpuInfo.uuid, gpuHaInfo.uuid().c_str(), DCGM_DEVICE_UUID_BUFFER_SIZE);

                gpuInfoList.push_back(gpuInfo);
            }

            // load each Switch information
            for ( int switchIdx = 0; switchIdx < partHaInfo.switchlist_size(); switchIdx++ ) {
                PartitionSwitchInfo switchInfo;
                const fabricmanagerHA::partitionSwitchInfo &swHaInfo = partHaInfo.switchlist(switchIdx);
                switchInfo.physicalId = swHaInfo.physicalid();
                switchInfo.numEnabledLinks = swHaInfo.numenabledlinks();
                switchInfo.enabledLinkMask = swHaInfo.enabledlinkmask();
                switchInfoList.push_back(switchInfo);
            }

            partInfo.partitionId = partHaInfo.partitionid();

            // the set activated partition list API processing will set isActive true
            // for the activated partitions.
            partInfo.isActive = false;
            partInfo.errorHandled = false;
            partInfo.gpuInfo = gpuInfoList;
            partInfo.switchInfo = switchInfoList;
            partInfo.trunkConnCount = partHaInfo.trunkconncount();
            mSupportedPartitions.push_back(partInfo);
        }
    }

    return true;
}


/*************************************************************************************
 * Interface to query lwrrently supported fabric partitions.
 * 
 * The interface can be used by higher level entities (like GFM) to get the list of 
 * available fabric partitions.
 * 
 **************************************************************************************/
dcgmReturn_t
DcgmGFMFabricPartitionMgr::getPartitions(dcgmFabricPartitionList_t &dcgmFabricPartitions)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    DcgmFMAutoLock lock(mLock); 

    PartitionInfoList::iterator it;
    int partIdx = 0;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo tempInfo = (*it);
        dcgmFabricPartitions.partitionInfo[partIdx].partitionId = tempInfo.partitionId;
        dcgmFabricPartitions.partitionInfo[partIdx].isActive = tempInfo.isActive;
        dcgmFabricPartitions.partitionInfo[partIdx].numGpus = tempInfo.gpuInfo.size();
        // copy all the GPU information
        PartitionGpuInfoList::iterator jit;
        int gpuIdx = 0;
        for (jit = tempInfo.gpuInfo.begin(); jit != tempInfo.gpuInfo.end(); jit++) {
            PartitionGpuInfo tempGpuInfo = (*jit);
            // Note: GPU physical Ids starts from 1 to external APIs. But, internally
            // the IDs starts from 0. So do a +1 to offset the difference
            dcgmFabricPartitions.partitionInfo[partIdx].gpuInfo[gpuIdx].physicalId = (tempGpuInfo.physicalId) + 1;
            strncpy(dcgmFabricPartitions.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid,
                    tempGpuInfo.uuid, sizeof(dcgmFabricPartitions.partitionInfo[partIdx].gpuInfo[gpuIdx].uuid));
            // fill PCI Bus ID information in LWML format
            DcgmFMPciInfo *pPciInfo = &tempGpuInfo.dynamicInfo.pciInfo;
            snprintf(dcgmFabricPartitions.partitionInfo[partIdx].gpuInfo[gpuIdx].pciBusId,
                     LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, LWML_DEVICE_PCI_BUS_ID_FMT,
                     LWML_DEVICE_PCI_BUS_ID_FMT_ARGS(pPciInfo));
            gpuIdx++;
        }
        partIdx++;
    }

    dcgmFabricPartitions.numPartitions = partIdx;
    return DCGM_ST_OK;
}

/*************************************************************************************
 * Interface to activate a partition
 * 
 * 
 **************************************************************************************/
dcgmReturn_t
DcgmGFMFabricPartitionMgr::activatePartition(uint32 nodeId, unsigned int partitionId)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    DcgmFMAutoLock lock(mLock); 
    FM_ERROR_CODE fmRetVal;

    if (!isPartitionExists(nodeId, partitionId)) {
        FM_SYSLOG_ERR("Specified partition id %d is not supported", partitionId);
        return DCGM_ST_BADPARAM;
    }

    if (isPartitionActive(nodeId, partitionId)) {
        FM_SYSLOG_ERR("Specified partition id %d is already activated", partitionId);
        return DCGM_ST_IN_USE;
    }

    if (isGpuUsedInActivePartitions(nodeId, partitionId)) {
        FM_SYSLOG_ERR("Specified partition id %d has GPUs that are already in use", partitionId);
        return DCGM_ST_IN_USE;
    }

    PartitionInfo partInfo;
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    //
    // request the localFM to disable required GPU LWLinks before initiating any
    // real GPU attach(aka initialization) and link training sequence.
    //
    fmRetVal = mGfm->mpConfig->configSetGpuDisabledLinkMaskForPartition(nodeId, partInfo);
    if (FM_SUCCESS != fmRetVal) {
        FM_SYSLOG_ERR("Failed to disable desired GPU LWLinks for partition id  %d", partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    // request the node's LocalFM to attach all the GPUs.
    fmRetVal = mGfm->mpConfig->configSharedLWSwitchPartitionAttachGPUs(nodeId, partitionId);
    if (FM_SUCCESS != fmRetVal) {
        FM_SYSLOG_ERR("Failed to attach GPUs for partition id %d", partitionId);
        handlePartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    // get attached GPU information from node's LocalFM
    DcgmFMGpuInfoMap gpuInfoMap;
    DcgmFMGpuInfoMap blacklistGpuInfoMap;
    DcgmGFMHelper::getGpuDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, gpuInfoMap, blacklistGpuInfoMap);
    // verify whether all the required GPUs in the partition is attached successfully
    if (!validatePartitionGpus(nodeId, partitionId, gpuInfoMap)) {
        FM_SYSLOG_ERR("Required GPUs are not attached or missing for partition id %d", partitionId);
        handlePartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    // get LWLink device information from LWLinkCoereLib driver.
    DcgmGFMLWLinkDevRepo lwLinkDevRepo;
    DcgmGFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    // do LWLink initialization for the node
    DcgmGFMHelper::lwLinkInitializeNode(nodeId, mGfm->mLinkTrainIntf, lwLinkDevRepo);

    // discover all the intra node connections
    DcgmFMLWLinkConnRepo lwLinkConnRepo;
    DcgmGFMHelper::lwLinkDiscoverIntraNodeConnOnNode(nodeId, mGfm->mLinkTrainIntf, lwLinkConnRepo);

    //
    // as part of connection discovery, LWLink trunk connections will be detected as well.
    // we don't have to train the trunk connections if this partition doesn’t require it.
    // so, remove the same from our connection list before starting the actual training
    //
    filterPartitionLWLinkTrunkConns(nodeId, partitionId, lwLinkConnRepo, lwLinkDevRepo);

    // train all connections
    DcgmGFMHelper::lwLinkTrainIntraNodeConnections(mGfm->mLinkTrainIntf, lwLinkConnRepo,
                                                   lwLinkDevRepo, LWLINK_TRAIN_SAFE_TO_HIGH);

    lwLinkConnRepo.dumpAllConnAndStateInfo(lwLinkDevRepo);

    // update GPU enum index in the partition for config
    updatePartitionGpuDynamicInfo(nodeId, partitionId, gpuInfoMap);

    // verify all the LWLink connections and its state.
    if (!validatePartitionLWLinkConns(nodeId, partitionId, lwLinkConnRepo)) {
        FM_SYSLOG_ERR("LWLink connections are missing or not trained for partition id %d", partitionId);
        handlePartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    //
    // all the pre-conditions are met, and partition information updated.
    // go ahead and configure GPUs and LWSwitches for this partition.
    // We need to update our locally cahed parition information with new
    // gpu enumeration index and bdfs.
    //
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);

    // enable routing table entries, configure GPU fabric address
    fmRetVal = mGfm->mpConfig->configActivateSharedLWSwitchPartition(nodeId, partInfo);
    if (FM_SUCCESS != fmRetVal) {
        FM_SYSLOG_ERR("Failed to configure LWSwitch/GPU for partition id %d", partitionId);
        handlePartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    // detach GPUs
    fmRetVal = mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(nodeId, partitionId);
    if (FM_SUCCESS != fmRetVal) {
        FM_SYSLOG_ERR("Failed to detach GPUs for partition id %d", partitionId);
        handlePartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    //
    // All the trunk connections/link will be in SAFE mode if the partition doesn’t require them
    // as we filtered and skipped training. reset those links to completely power them off
    //
    partitionResetFilteredTrunkLWLinks(nodeId, partitionId, false);

    // request messages are async, the errors could have happened when
    // response messages are received in a different thread.
    // Before return to the API caller, check if any error has oclwrred
    // and handled in the other thread.
    if (isPartitionConfigFailed(nodeId, partitionId))
    {
        // Clear pending request before starting next partition activation
        mGfm->mpConfig->clearPartitionPendingConfigRequest(nodeId, partitionId);
        clearPartitionConfigFailure(nodeId, partitionId);
        return DCGM_ST_GENERIC_ERROR;
    }

    // mark the partition state as active
    setPartitionActiveState(nodeId, partitionId, true);

    PRINT_INFO("%d", "Partition id %d is activated.", partitionId);
    FM_SYSLOG_NOTICE("Partition id %d is activated.", partitionId);
    return DCGM_ST_OK;
}

/*************************************************************************************
 * Interface to deactivate a partition
 * 
 * 
 **************************************************************************************/
dcgmReturn_t
DcgmGFMFabricPartitionMgr::deactivatePartition(uint32 nodeId, unsigned int partitionId)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    DcgmFMAutoLock lock(mLock); 
    FM_ERROR_CODE fmRetVal;
    dcgmReturn_t dcgmReturn;

    if (!isPartitionExists(nodeId, partitionId)) {
        FM_SYSLOG_ERR("Specified partition id %d is not supported", partitionId);
        return DCGM_ST_BADPARAM;
    }

    if (!isPartitionActive(nodeId, partitionId)) {
        FM_SYSLOG_ERR("Specified partition id %d is not activated ", partitionId);
        return DCGM_ST_UNINITIALIZED;
    }

    PartitionInfo partInfo;
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo);
    dcgmReturn = DCGM_ST_OK;

    // reset LWSwitch side lwlinks used in this partition
    // the reset links has to happen before disabling routing to prevent any ACL error
    // as the GPU/GuestVM may be sending traffic or in-flight traffic.
    if (!resetPartitionLWSwitchLinks(nodeId, partitionId, false)) {
        FM_SYSLOG_ERR("Failed to reset LWSwitch side links for partition id %d", partitionId);
        dcgmReturn = DCGM_ST_GENERIC_ERROR;
        // fall through to disable routing next
    }

    // disable routing table entries even reset lwlinks is not successfull
    fmRetVal = mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, partInfo);
    if (FM_SUCCESS != fmRetVal) {
        FM_SYSLOG_ERR("Failed to deconfigure LWSwitch/GPU for partition id %d", partitionId);
        dcgmReturn = DCGM_ST_GENERIC_ERROR;
        // fall through to clear error states next
    }

    // mark the partition state as inactive
    setPartitionActiveState(nodeId, partitionId, false);

    PRINT_INFO("%d", "Partition id %d is deactivated.", partitionId);
    FM_SYSLOG_NOTICE("Partition id %d is deactivated.", partitionId);

    // request messages are async, the errors could have happened when
    // response messages are received in a different thread.
    // Before return to the API caller, check if any error has oclwrred
    // and handled in the other thread.
    if (isPartitionConfigFailed(nodeId, partitionId) || (dcgmReturn != DCGM_ST_OK) )
    {
        // Clear pending request before starting next partition deactivation
        mGfm->mpConfig->clearPartitionPendingConfigRequest( nodeId, partitionId );
        clearPartitionConfigFailure(nodeId, partitionId);
        dcgmReturn = DCGM_ST_GENERIC_ERROR;
    }

    return dcgmReturn;
}

/*************************************************************************************
 * Interface to set a list of already activated partitions after restart
 **************************************************************************************/
dcgmReturn_t
DcgmGFMFabricPartitionMgr::setActivatedPartitions(dcgmActivatedFabricPartitionList_t &dcgmFabricPartitions)
{
    // serialize FM LWSwitch shared virtualization related APIs.
    DcgmFMAutoLock lock(mLock);
    FM_ERROR_CODE fmRetVal;

    uint32_t nodeId = 0, i, partitionId;
    PartitionInfo partInfo;

    PRINT_INFO("%d", "setActivatedPartitions number of partitions %d.",
               dcgmFabricPartitions.numPartitions);

    for (i = 0; i < dcgmFabricPartitions.numPartitions; i++) {

        partitionId = dcgmFabricPartitions.partitionIds[i];

        if ( getSharedLWSwitchPartitionInfo(0, partitionId, partInfo) == false ) {
            FM_SYSLOG_ERR("Specified partition id %d is not supported", partitionId);
            return DCGM_ST_BADPARAM;
        }

        // mark the partition state as active
        PRINT_INFO("%d", "Partition id %d is activated.", partitionId);
        setPartitionActiveState(nodeId, partitionId, true);
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
        if (tempInfo.isActive == true) {
            continue;
        }

        if (isGpuUsedInActivePartitions(nodeId, tempInfo.partitionId)) {
            // disabled partition use the same GPUs in the activated partitions
            continue;
        }

        partitionId = tempInfo.partitionId;
        // the partition is not activated, disable routing entries
        fmRetVal = mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(nodeId, tempInfo);
        if (FM_SUCCESS != fmRetVal) {
            FM_SYSLOG_ERR("Failed to deconfigure LWSwitch/GPU for partition id %d", partitionId);
            return DCGM_ST_GENERIC_ERROR;
        }

        PRINT_DEBUG("%d", "Partition id %d is deactivated.", partitionId);

        // request messages are async, the errors could have happened when
        // response messages are received in a different thread.
        // Before return to the API caller, check if any error has oclwrred
        // and handled in the other thread.
        if (isPartitionConfigFailed(nodeId, partitionId)) {
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    mInitDone = true;
    return DCGM_ST_OK;
}

bool
DcgmGFMFabricPartitionMgr::getSharedLWSwitchPartitionInfo(uint32 nodeId, unsigned int partitionId,
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
DcgmGFMFabricPartitionMgr::mapTopologyPartition(uint32 nodeId, const sharedLWSwitchPartitionInfo &topoPartInfo)
{
    PartitionInfo partInfo;
    PartitionSwitchInfoList switchInfoList;
    PartitionGpuInfoList gpuInfoList;

    switchInfoList.clear();
    gpuInfoList.clear();

    // check whether we already have a partition with specified id
    if (getSharedLWSwitchPartitionInfo(nodeId, topoPartInfo.partitionid(), partInfo)) {
        PRINT_ERROR("%d", "partition id %d already exists while creating a partition map",
                     topoPartInfo.partitionid());
        return false;
    }

    // validate whether we detected the required number of GPUs and LWSwitches
    uint32_t requiredGpuCount = topoPartInfo.gpuinfo_size();
    uint32_t requiredSwitchCount =  topoPartInfo.switchinfo_size();

    DcgmFMGpuInfoList detectedGpuInfo = mGfm->mGpuInfoMap[nodeId];
    uint32_t detectedGpuCount = detectedGpuInfo.size();

    DcgmFMLWSwitchInfoList detectedSwitchInfo = mGfm->mLwswitchInfoMap[nodeId];
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

        gpuInfoList.push_back(gpuInfo);        
    }

    // copy each Switch information
    for ( int switchIdx = 0; switchIdx < topoPartInfo.switchinfo_size(); switchIdx++ ) {
        PartitionSwitchInfo switchInfo;
        const sharedLWSwitchPartitionSwitchInfo &topoSwitchInfo = topoPartInfo.switchinfo(switchIdx);
        switchInfo.physicalId = topoSwitchInfo.physicalid();
        switchInfo.numEnabledLinks = topoSwitchInfo.numenabledlinks();
        switchInfo.enabledLinkMask = topoSwitchInfo.enabledlinkmask();
        switchInfoList.push_back(switchInfo);
    }

    // create the full partition information and save
    partInfo.partitionId = topoPartInfo.partitionid();
    partInfo.isActive = false;
    partInfo.gpuInfo = gpuInfoList;
    partInfo.switchInfo = switchInfoList;
    partitionMetaDataInfo partMetaData = topoPartInfo.metadata();
    partInfo.trunkConnCount = partMetaData.lwlinkintratrunkconncount();
    mSupportedPartitions.push_back(partInfo);

    return true;
}

bool
DcgmGFMFabricPartitionMgr::isPartitionExists(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfo tempPartInfo;
    return getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo);
}

void
DcgmGFMFabricPartitionMgr::setPartitionActiveState(uint32 nodeId, unsigned int partitionId, bool active)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.isActive = active;
            break;
        }
    }
}

bool
DcgmGFMFabricPartitionMgr::isPartitionActive(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfo tempPartInfo = {0};
    // no need to check return value as we will return false even if the partition is not found
    getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo);
    return tempPartInfo.isActive;
}

// put all GPU pysicalIds used in already activated partitions in std::set
void
DcgmGFMFabricPartitionMgr::getGpusUsedInActivatePartitions(uint32_t nodeId, std::set<uint32_t> &usedGpus)
{
    usedGpus.clear();

    PartitionInfoList::iterator partIt;
    for (partIt = mSupportedPartitions.begin(); partIt != mSupportedPartitions.end(); ++partIt) {
        PartitionInfo &partInfo = *partIt;
        if (partInfo.isActive == false)
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
DcgmGFMFabricPartitionMgr::isGpuUsedInActivePartitions(uint32_t nodeId, uint32_t partitionId)
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
            PRINT_DEBUG("%d", "find gpu %d is used.", gpuInfo.physicalId);
            return true;
        }
    }

    return false;
}

bool
DcgmGFMFabricPartitionMgr::validatePartitionGpus(uint32 nodeId, unsigned int partitionId,
                                                 DcgmFMGpuInfoMap &gpuInfoMap)
{
    PartitionInfo tempPartInfo;
    DcgmFMGpuInfoMap::iterator it = gpuInfoMap.find(nodeId);
    DcgmFMGpuInfoList detectedGpuInfoList = it->second;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, tempPartInfo)) {
        return false;
    }

    // validate total number of GPUs
    if (tempPartInfo.gpuInfo.size() != detectedGpuInfoList.size()) {
        PRINT_ERROR("%d %ld %ld", "Missing GPUs in partition Id %d, expected GPUs %ld detected GPUs %ld",
                    partitionId, tempPartInfo.gpuInfo.size(), detectedGpuInfoList.size());
        return false;
    }

    // validate each GPU by comparing their UUIDs. This will be persistent across
    // each attach/detach cycle by hypervisor.
    PartitionGpuInfoList::iterator jit;
    for (jit = tempPartInfo.gpuInfo.begin(); jit != tempPartInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo partGpuInfo = (*jit);
        bool bFound = false;
        DcgmFMGpuInfoList::iterator git;
        for (git = detectedGpuInfoList.begin(); git != detectedGpuInfoList.end(); git++) {
            DcgmFMGpuInfo fmGpuInfo = (*git);
            if (strncasecmp(partGpuInfo.uuid, fmGpuInfo.uuid, sizeof(partGpuInfo.uuid)) == 0) {
                bFound = true;
                break;
            }
        }

        // check whether we found the specified GPU
        if (bFound == false) {
            PRINT_ERROR("%s %d", "GPU UUID:%s is missing or not attached for partition id %d",
                        partGpuInfo.uuid, partitionId);
            return false;
        }
    }

    // all the validation passed.
    return true;
}

bool
DcgmGFMFabricPartitionMgr::validatePartitionLWLinkConns(uint32 nodeId, unsigned int partitionId,
                                                        DcgmFMLWLinkConnRepo &lwLinkConnRepo)

{
    PartitionInfo tempPartInfo;
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
        if (detectedTrunkConnCnt != tempPartInfo.trunkConnCount) {
            PRINT_ERROR("%d %d %d", "partition id %d requires %d trunk connections, detected %d connections",
                        partitionId, tempPartInfo.trunkConnCount, detectedTrunkConnCnt);
            return false;
        }
        if (!mGfm->mTopoValidator->isIntraNodeTrunkConnsActive(nodeId, lwLinkConnRepo)) {
            PRINT_ERROR("%d", "all the trunk connections are not trained to ACTIVE for partition id %d",
                        partitionId);
            return false;
        }
    }

    // check whether all the access connections are detected and trained to high speed
    uint32 detectedAccessConnCnt = mGfm->mTopoValidator->getAccessConnCount(nodeId, lwLinkConnRepo);

    // compute total access connection count expected
    uint32 accessConnCnt = 0;
    PartitionGpuInfoList::iterator jit;
    for (jit = tempPartInfo.gpuInfo.begin(); jit != tempPartInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo partGpuInfo = (*jit);
        accessConnCnt += partGpuInfo.numEnabledLinks;
    }

    if (detectedAccessConnCnt != accessConnCnt) {
        PRINT_ERROR("%d %d %d", "partition id %d requires %d access connections, detected %d connections",
                        partitionId, accessConnCnt, detectedAccessConnCnt);
        return false;
    }
    if (!mGfm->mTopoValidator->isAccessConnsActive(nodeId, lwLinkConnRepo)) {
        PRINT_ERROR("%d", "all the access connections are not trained to ACTIVE for partition id %d",
                    partitionId);
        return false;
    }

    // all the validation looks good
    return true;
}

void
DcgmGFMFabricPartitionMgr::updatePartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId,
                                                         DcgmFMGpuInfoMap &gpuInfoMap)
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

    DcgmFMGpuInfoMap::iterator gpuMapit = gpuInfoMap.find(nodeId);
    DcgmFMGpuInfoList detectedGpuInfoList = gpuMapit->second;
    // now update each GPU's dynamic information by comparing their UUIDs
    PartitionGpuInfoList::iterator jit;
    for (jit = partInfo.gpuInfo.begin(); jit != partInfo.gpuInfo.end(); jit++) {
        PartitionGpuInfo &partGpuInfo = (*jit);
        // find the corresponding GPU from the lwrrently attached GPUs.
        DcgmFMGpuInfoList::iterator git;
        for (git = detectedGpuInfoList.begin(); git != detectedGpuInfoList.end(); git++) {
            DcgmFMGpuInfo fmGpuInfo = (*git);
            if (strncasecmp(partGpuInfo.uuid, fmGpuInfo.uuid, sizeof(partGpuInfo.uuid)) == 0) {
                partGpuInfo.dynamicInfo.gpuIndex = fmGpuInfo.gpuIndex;
                partGpuInfo.dynamicInfo.pciInfo = fmGpuInfo.pciInfo;
                break;
            }
        }
   }
}

bool
DcgmGFMFabricPartitionMgr::resetPartitionLWSwitchLinks(uint32 nodeId, unsigned int partitionId,
                                                       bool inErrHdlr)
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
    partitionPowerOffTrunkLWLinkConns(nodeId, partitionId, inErrHdlr);

    // reset corresponding LWSwitch links.
    PartitionSwitchInfoList::iterator it;
    for (it = partInfo.switchInfo.begin(); it != partInfo.switchInfo.end(); it++ ) {
        PartitionSwitchInfo switchInfo = (*it);
        // reset should be in pairs. so compute the mask considering the odd/even pair
        uint64 tempEnabledMask = switchInfo.enabledLinkMask;
        uint64 resetLinkMask = 0;
        for (uint64_t linkId = 0; tempEnabledMask != 0; linkId +=2, tempEnabledMask >>= 2) {
            if ((tempEnabledMask & 0x3) != 0) {
                // rebuild the mask
                resetLinkMask |= (BIT64(linkId) | BIT64(linkId + 1));
            }
        }

        // go ahead and reset the links used for this partition
        int tempRet = DcgmGFMHelper::lwLinkSendResetSwitchLinks(nodeId, switchInfo.physicalId,
                                                                resetLinkMask, mGfm->mLinkTrainIntf,
                                                                inErrHdlr);
        if (tempRet) {
            PRINT_ERROR("%d", "failed to do LWLink reset for LWSwitch id %d ", switchInfo.physicalId );
            // indicate the overall status, but continue with other switches.
            retVal = false;
        }
    }

    return retVal;
}

void
DcgmGFMFabricPartitionMgr::filterPartitionLWLinkTrunkConns(uint32 nodeId, unsigned int partitionId,
                                                           DcgmFMLWLinkConnRepo &linkConnRepo,
                                                           DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    PartitionInfo partInfo;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount != 0) {
        // no trunk connection to remove, ie 16 GPU VM on DGX-2/HGX-2 
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
    DcgmLWLinkDetailedConnList tempIntraConnList = it->second;
    DcgmLWLinkDetailedConnList::iterator connIt;

    // iterate over all the connections in the node and remove trunk connections
    for (connIt = tempIntraConnList.begin(); connIt != tempIntraConnList.end();) {
        DcgmFMLWLinkDetailedConnInfo *lwLinkConn = *connIt;
        if (partitionIsLWLinkTrunkConnection(lwLinkConn, linkDevRepo)) {
            // remove this connection
            connIt = tempIntraConnList.erase( connIt );
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
DcgmGFMFabricPartitionMgr::partitionIsLWLinkTrunkConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                                            DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    DcgmLWLinkEndPointInfo endPoint0 = lwLinkConn->getMasterEndPointInfo();
    DcgmLWLinkEndPointInfo endPoint1 = lwLinkConn->getSlaveEndPointInfo();
    DcgmFMLWLinkDevInfo end0LWLinkDevInfo; 
    DcgmFMLWLinkDevInfo end1LWLinkDevInfo;

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
DcgmGFMFabricPartitionMgr::partitionResetFilteredTrunkLWLinks(uint32 nodeId,
                                                              unsigned int partitionId,
                                                              bool inErrHdlr)
{
    PartitionInfo partInfo;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount != 0) {
        // all the trunk connections are required and nothing to reset ie 16 GPU VM on DGX-2/HGX-2 
        return;
    }

    // For DGX-2 and HGX-2, trunk links are 0,1,2,3,8,9,10,11 for all the LWSwitches.
    // Note/TODO: Change this mask based on platform/configurable.
    uint64 resetLinkMask = 0xF0F;

    //
    // all the trunk links are in safe mode now and we need to reset across all the
    // switches. so taking the switches from partition context is not enough
    // so go through all the detected switches for the node.
    //
    // Note/TODO: Make this platform agnostic/configurable.
    //
    DcgmFMLWSwitchInfoMap::iterator it = mGfm->mLwswitchInfoMap.find(nodeId);
    if (it == mGfm->mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return;
    }

    DcgmFMLWSwitchInfoList switchList = it->second;
    DcgmFMLWSwitchInfoList::iterator jit;
    DcgmFMLWSwitchInfo switchInfo = switchList.front();
    if (!(switchInfo.enabledLinkMask & resetLinkMask)) {
        // trunk links are already disabled as part of multi-host. no need to do reset
        return;
    }

    // go ahead and reset each trunk links.
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        DcgmFMLWSwitchInfo switchInfo = (*jit);
        // go ahead and reset the trunk links
        int tempRet = DcgmGFMHelper::lwLinkSendResetSwitchLinks(nodeId, switchInfo.physicalId,
                                                                resetLinkMask, mGfm->mLinkTrainIntf,
                                                                inErrHdlr);
        if (tempRet) {
            PRINT_ERROR("%d", "failed to do LWLink trunk link reset for LWSwitch id %d ", switchInfo.physicalId );
            // best effort, just log the error and continue.
        }
    }
}

void
DcgmGFMFabricPartitionMgr::partitionPowerOffTrunkLWLinkConns(uint32 nodeId,
                                                             unsigned int partitionId,
                                                             bool inErrHdlr)
{
    PartitionInfo partInfo;
    int retVal = true;

    // get the specified partition information
    if (!getSharedLWSwitchPartitionInfo(nodeId, partitionId, partInfo)) {
        return;
    }

    if (partInfo.trunkConnCount == 0) {
        // no trunk connections used for this partition, ie not a 16 GPU VM on DGX-2/HGX-2 
        return;
    }

    //
    // instead of caching the trunk connections, we are querying it again from the driver.
    //

    // get LWLink device information from LWLinkCoereLib driver.
    DcgmGFMLWLinkDevRepo lwLinkDevRepo;
    DcgmGFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    // get the list of all the intra node connections
    DcgmFMLWLinkConnRepo lwLinkConnRepo;
    DcgmGFMHelper::lwLinkGetIntraNodeConnOnNodes(nodeId, mGfm->mLinkTrainIntf, lwLinkConnRepo);

    // reset all the trunk connections.
    // Note: No need to filter the detected connections as we should only get trunk connections
    // as the GPU are detached from service VM and associated access connections are removed.
    DcgmGFMHelper::lwLinkTrainIntraNodeConnections(mGfm->mLinkTrainIntf, lwLinkConnRepo,
                                                   lwLinkDevRepo, LWLINK_TRAIN_TO_OFF, inErrHdlr);
}

void
DcgmGFMFabricPartitionMgr::handlePartitionConfigFailure(uint32 nodeId, unsigned int partitionId)
{
    lwswitch::fmMessage errMsg;
    mGfm->mpConfig->handleSharedLWSwitchPartitionConfigError(nodeId, partitionId,
                                                             ERROR_SOURCE_SW_GLOBALFM,
                                                             ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED,
                                                             errMsg);
}

void
DcgmGFMFabricPartitionMgr::clearPartitionConfigFailure(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.errorHandled = false;
            break;
        }
    }
}

void
DcgmGFMFabricPartitionMgr::setPartitionConfigFailure(uint32 nodeId, unsigned int partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            tempInfo.errorHandled = true;
            break;
        }
    }
}

bool
DcgmGFMFabricPartitionMgr::isPartitionConfigFailed(uint32 nodeId, uint32_t partitionId)
{
    PartitionInfoList::iterator it;
    for (it = mSupportedPartitions.begin(); it != mSupportedPartitions.end(); it++) {
        PartitionInfo &tempInfo = (*it);
        if (tempInfo.partitionId == partitionId) {
            return tempInfo.errorHandled;
        }
    }

    return false;
}

// Get the first active partition id for the given LWSwitch and PortNumber pair
// if found, a valid partitionId is returned
// if not found, ILWALID_FABRIC_PARTITION_ID is returned
uint32_t
DcgmGFMFabricPartitionMgr::getActivePartitionIdForLWSwitchPort(uint32_t nodeId,
                                                               uint32_t physicalId,
                                                               uint32_t portNum)
{
    PartitionInfoList::iterator partIt;
    for (partIt = mSupportedPartitions.begin();
         partIt != mSupportedPartitions.end();
         ++partIt) {

        PartitionInfo &partInfo = *partIt;
        if (partInfo.isActive == 0) continue;

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
