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
#include <iomanip>

#include "fm_log.h"
#include "FMAutoLock.h"
#include "FMErrorCodesInternal.h"
#include "GlobalFabricManager.h"
#include "GlobalFmFabricParser.h"
#include "GFMFabricPartitionMgr.h"
#include "GlobalFmDegradedModeMgr.h"
#include "GlobalFmApiHandler.h"
#include "GFMHelper.h"
#include "ctrl/ctrl2080.h"

GlobalFmApiHandler::GlobalFmApiHandler(GlobalFabricManager *pGfm)
{
    mpGfm = pGfm;

    lwosInitializeCriticalSection( &mLock );
}

GlobalFmApiHandler::~GlobalFmApiHandler()
{
    lwosDeleteCriticalSection( &mLock );
}

// process fmlib API command FM_GET_SUPPORTED_PARTITIONS
FMIntReturn_t
GlobalFmApiHandler::getSupportedFabricPartitions(fmFabricPartitionList_t &fmFabricPartitionList)
{
    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch and vGPU based multitenancy modes.
    if ((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) {
        FM_LOG_WARNING("get fabric partitions called while not in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        // not logging any error message here as this may be called repeatedly to check the
        // fabric manager initialization state by Hypervisor
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if ((mpGfm->mpHaMgr == NULL) || (mpGfm->mpHaMgr->isInitDone() == false)) {
        // not logging any error message here as this may be called repeatedly to check the
        // fabric manager initialization state by Hypervisor
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // all looks good, ilwoke the actual partition manager API.
    return mpGfm->mGfmPartitionMgr->getPartitions(fmFabricPartitionList);
}

// process fmlib API command FM_ACTIVATE_PARTITION
FMIntReturn_t
GlobalFmApiHandler::activateFabricPartition(fmFabricPartitionId_t partitionId)
{
    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch multitenancy mode.
    if (mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) {
        FM_LOG_WARNING("activate fabric partition called while not in Shared LWSwitch multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        FM_LOG_WARNING("activate fabric partition called before initialization is completed");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if ((mpGfm->mpHaMgr == NULL) || (mpGfm->mpHaMgr->isInitDone() == false)) {
        FM_LOG_WARNING("activate fabric partition called before saved fabric state is reloaded");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual partition manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    return mpGfm->mGfmPartitionMgr->activatePartition(0, partitionId);
}

// process fmlib API command FM_ACTIVATE_PARTITION_WITH_VFS
FMIntReturn_t
GlobalFmApiHandler::activateFabricPartitionWithVfs(fmFabricPartitionId_t partitionId, fmPciDevice_t *vfList, unsigned int numVfs)
{
    FMIntReturn_t rc = FM_INT_ST_OK;

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in shared fabric mode.
    if (mpGfm->getFabricMode() != FM_MODE_VGPU) {
        FM_LOG_WARNING("activate fabric partition called while not in vGPU multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        FM_LOG_WARNING("activate fabric partition called before initialization is completed");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if ((mpGfm->mpHaMgr == NULL) || (mpGfm->mpHaMgr->isInitDone() == false)) {
        FM_LOG_WARNING("activate fabric partition called before saved fabric state is reloaded");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual partition manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    rc = mpGfm->mGfmPartitionMgr->activatePartitionWithVfs(0, partitionId, vfList, numVfs);

    // save fm state only if partition activation is successful
    if (rc == FM_INT_ST_OK) {
        mpGfm->mpHaMgr->saveStates();
    }

    return rc; 
}

// process fmlib API command FM_DEACTIVATE_PARTITION
FMIntReturn_t
GlobalFmApiHandler::deactivateFabricPartition(fmFabricPartitionId_t partitionId)
{
    FMIntReturn_t rc = FM_INT_ST_OK;

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch and vGPU based multitenancy modes.
    if ((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) {
        FM_LOG_WARNING("deactivate partition called while not in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        FM_LOG_WARNING("deactivate fabric partition called before initialization is completed");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if ((mpGfm->mpHaMgr == NULL) || (mpGfm->mpHaMgr->isInitDone() == false)) {
        FM_LOG_WARNING("deactivate fabric partition called before saved fabric state is reloaded");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual partition manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    rc = mpGfm->mGfmPartitionMgr->deactivatePartition(0, partitionId);

    // save fm state only for vGPU mode
    if (mpGfm->getFabricMode() == FM_MODE_VGPU) {
        mpGfm->mpHaMgr->saveStates();
    }

    return rc;
}

// process fmlib API command FM_SET_ACTIVATED_PARTITION_LIST
FMIntReturn_t
GlobalFmApiHandler::setActivatedFabricPartitions(fmActivatedFabricPartitionList_t &fmFabricPartitions)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch and vGPU based multitenancy modes and plus, when FM is restarted.
    if (((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) || (mpGfm->isFabricModeRestart() == false)) {
        FM_LOG_WARNING("set activated fabric partition called while not in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        FM_LOG_WARNING("set activated fabric partition called before initialization is completed");
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // this API should only be called before HA mgr init done
    if ((mpGfm->mpHaMgr != NULL) && (mpGfm->mpHaMgr->isInitDone() == true)) {
        FM_LOG_WARNING("set activated fabric partition called after saved fabric state is reloaded");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    //
    // Note: no need to wait for HaMgr completes HA initialization
    // because this API is part of the HA restart process
    //

    // set the activated partition list provided by hypervisor after fm restart
    rc = mpGfm->mGfmPartitionMgr->setActivatedPartitions(fmFabricPartitions);

    // save fm state only for vGPU mode
    if ((mpGfm->getFabricMode() == FM_MODE_VGPU) && (rc == FM_INT_ST_OK)) {
        mpGfm->mpHaMgr->saveStates();
    }

    // notify the HAMgr to update its state
    mpGfm->mpHaMgr->updateHaInitDoneState();

    return rc;
}

// process fmlib API command FM_GET_LWLINK_FAILED_DEVICES
FMIntReturn_t
GlobalFmApiHandler::getLwlinkFailedDevices(fmLwlinkFailedDevices_t &devList)
{
    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch and vGPU based multitenancy modes and plus, when FM is restarted.
    if ((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) {
        FM_LOG_WARNING("get LWLink failed devices called while not in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    if (mpGfm->isFabricModeRestart() == true) {
        FM_LOG_WARNING("get LWLink failed devices is not supported when Fabric Manager is in restart mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        // not logging any error message here as this may be called repeatedly to check the
        // fabric manager initialization state by Hypervisor
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after degraded mode manager is initialized
    if (mpGfm->mpDegradedModeMgr == NULL) {
        // not logging any error message here as this may be called repeatedly
        return FM_INT_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual degrade manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    mpGfm->mpDegradedModeMgr->getLwlinkFailedDevices(0, &devList);
    return FM_INT_ST_OK;
}

// process fmlib API command FM_GET_UNSUPPORTED_PARTITIONS
FMIntReturn_t
GlobalFmApiHandler::getUnsupportedFabricPartitions(fmUnsupportedFabricPartitionList_t &fmFabricPartitionList)
{
    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // these APIs are supported only in Shared LWSwitch and vGPU based multitenancy modes.
    if ((mpGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) && (mpGfm->getFabricMode() != FM_MODE_VGPU)) {
        FM_LOG_WARNING("get unsupported fabric partitions called while not in Shared LWSwitch or vGPU based multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        // not logging any error message here as this may be called repeatedly to check the
        // fabric manager initialization state by Hypervisor
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if ((mpGfm->mpHaMgr == NULL) || (mpGfm->mpHaMgr->isInitDone() == false)) {
        // not logging any error message here as this may be called repeatedly to check the
        // fabric manager initialization state by Hypervisor
        return FM_INT_ST_NOT_CONFIGURED;
    }

    // all looks good, ilwoke the actual partition manager API.
    return mpGfm->mGfmPartitionMgr->getUnsupportedPartitions(fmFabricPartitionList);
}

//
// process fm internal API command FM_PREPARE_GPU_FOR_RESET
//
// Step 1 from LWML for GPU reset process. FM will detach the GPU
// so that all the FM RM handles to the GPU will be closed.
//
FMIntReturn_t
GlobalFmApiHandler::prepareGpuForReset(char *gpuUuid)
{
    FMIntReturn_t rc;

    rc = isGpuResetSupported(gpuUuid);
    if ( rc != FM_INT_ST_OK) {
        return rc;
    }

    FM_LOG_INFO("processing request to prepare GPU uuid %s for reset", gpuUuid);

    rc = prepareGpuForResetHelper(gpuUuid);

    if (rc != FM_INT_ST_OK)
        return rc;
        
    //
    // add gpu to ResetGPUList
    //
    // For vGPU mode, we need to check and return GPU partition activation
    // if any GPU in the partition is in reset.
    // 
    if (mpGfm->getFabricMode() == FM_MODE_VGPU) {
        mpGfm->mGfmPartitionMgr->addGpuToResetPendingList(gpuUuid);
    }

    return FM_INT_ST_OK;
} 

FMIntReturn_t
GlobalFmApiHandler::prepareGpuForResetHelper(char *gpuUuid)
{
    uint32_t nodeId = 0; // API does not have nodeId, use 0 now.
    FMIntReturn_t rc;

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // Detach the GPU
    rc = mpGfm->mpConfig->configDetachGpu(nodeId, gpuUuid);
    if ( rc != FM_INT_ST_OK ) {
        // failed to send the detach request
        // error is already logged in the path
        return rc;
    }

    return FM_INT_ST_OK;
}

//
// process fm internal API command FM_SHUTDOWN_GPU_LWLINK
// Step 2 from LWML for GPU reset process. FM will train off all the Access LWLink
// connections of the specified GPU.
// LWML will make sure that at least one open handle (an auxiliary file descriptor) to the 
// GPU, so that the device and LWLink connections are present in LWLinkCoreLib Driver.
//
FMIntReturn_t
GlobalFmApiHandler::shutdownGpuLwlink(char *gpuUuid)
{
    uint32_t nodeId = 0; // API does not have nodeId, use 0 now.

    FMIntReturn_t rc = isGpuResetSupported(gpuUuid);
    if ( rc != FM_INT_ST_OK) {
        return rc;
    }

    FM_LOG_INFO("processing request to shutdown/train off GPU uuid %s LWLink connections", gpuUuid);

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // get LWLink device information from LWLinkCoereLib driver.
    GlobalFMLWLinkDevRepo lwLinkDevRepo;
    GFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mpGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    FMGpuInfo_t gpuInfo;
    // this should succeed as it is already validated in isGpuResetSupported
    mpGfm->getGpuInfo(gpuUuid, gpuInfo);

    // do the lookup on devices returned LWLinkCoreLib driver.
    FMLWLinkDevInfo lwLinkDevInfo;
    if (lwLinkDevRepo.getDeviceInfo(nodeId, gpuInfo.pciInfo, lwLinkDevInfo) == false ) {
        //
        // this can happen if a MIG enabled GPU's reset is requested. Lwrrenly LWLinks are disabled
        // for MIG enabled GPUs. In that case, FM has no access connections to train off.
        //
        FM_LOG_INFO("unable to get LWLink device information for GPU uuid %s. LWLinks may have been disabled for this GPU",
                    gpuUuid);
        return FM_INT_ST_OK;
    }

    // get new conn repo in case of degraded switches
    bool isDegraded = mpGfm->mpDegradedModeMgr->isAnyDeviceDegraded();
    GlobalFMLWLinkConnRepo lwLinkConnRepo;
    if (isDegraded) {
        //
        // fetch LWLink device and connections again as, if a device is degraded, it will be
        // unregistered from LWLinkCoreLib Driver and that will remove already detected 
        // and trained connections.
        //
        GFMHelper::lwLinkGetIntraNodeConnOnNodes(nodeId, mpGfm->mLinkTrainIntf, lwLinkConnRepo);
    }

    //
    // this GPU is found in LWLink driver context. It should have LWLinks active and train them off
    // in order to get the desried connections to train off, we need to first get the corresponding
    // device id used by LWLinkCoreLib driver.
    //
    uint64_t gpuLWLinkDrverId;
    if (mpGfm->getGpuLWLinkDriverId(nodeId, gpuUuid, gpuLWLinkDrverId) == false ) {

        FM_LOG_ERROR("unable to find GPU uuid %s information in LWLink driver context", gpuUuid);
        return FM_INT_ST_ILWALID_GPU;
    }

    // get only the access connections belonging to this gpu
    GlobalFMLWLinkConnRepo linkConnRepoByDriverId;
    GFMHelper::getLWLinkConnsRepoByLWLinkDriverId(nodeId, gpuLWLinkDrverId,
                                                  isDegraded ? lwLinkConnRepo : mpGfm->mLWLinkConnRepo,linkConnRepoByDriverId);

    // Train all the GPU access connections to OFF
    int ret;
    if ( mpGfm->isParallelTrainingEnabled() ) {
        ret = GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(mpGfm, mpGfm->mLinkTrainIntf,
                                                                 linkConnRepoByDriverId,
                                                                 isDegraded ? lwLinkDevRepo : mpGfm->mLWLinkDevRepo,
                                                                 LWLINK_TRAIN_TO_OFF);
    } else {
        ret = GFMHelper::lwLinkTrainIntraNodeConnections(mpGfm, mpGfm->mLinkTrainIntf,
                                                         linkConnRepoByDriverId,
                                                         isDegraded ? lwLinkDevRepo : mpGfm->mLWLinkDevRepo,
                                                         LWLINK_TRAIN_TO_OFF);
    }

    if (ret) {
        //
        // best effort to train off, just log the error and continue. The GPU and switch port
        // will get reset in next step.
        //
        FM_LOG_WARNING("failed to shutdown/train off LWLink connections for GPU uuid %s", gpuUuid);
    }

    return FM_INT_ST_OK;
}

//
// process fm internal API command FM_RESET_GPU_LWLINK
// Step 3 from LWML for GPU reset process. FM will reset and drain corresponding
// switch access ports
// By this time, LWML will close all the GPU open handles. So, the GPU and LWLink connections
// are not present in LWLinkCoreLib Driver.
//
FMIntReturn_t
GlobalFmApiHandler::resetGpuLwlink(char *gpuUuid)
{
    uint32_t nodeId = 0; // API does not have nodeId, use 0 now.
    FMIntReturn_t rc;

    rc = isGpuResetSupported(gpuUuid);
    if ( rc != FM_INT_ST_OK) {
        return rc;
    }

    FM_LOG_INFO("processing request to reset LWSwitch access LWLinks for GPU uuid %s", gpuUuid);

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    FMGpuInfo_t gpuInfo;
    // this should succeed as it is already validated in isGpuResetSupported
    mpGfm->getGpuInfo(gpuUuid, gpuInfo);

    //
    // if we found physical id of the GPU in our conext, it's links were active and we mapped it. 
    // and in above step, we trained them to off. Now reset and drain corresponding LWSwitch ports.
    // if we don't found it, that means the Links are not active (like MIG enabled GPU reset) and
    // we don't have to reset the Switch ports.
    //
    GpuKeyType gpuKey;
    gpuKey.nodeId = nodeId;
    if ( mpGfm->getGpuPhysicalId(nodeId, gpuUuid, gpuKey.physicalId) == false ) {
        FM_LOG_INFO("no LWLink mapping information for GPU uuid %s. LWLinks may have been disabled for this GPU", gpuUuid);
        return FM_INT_ST_OK;
    }

    // get all switch portMasks that connect to this GPU
    std::map <SwitchKeyType, uint64_t> connectedSwitches;
    mpGfm->mpParser->getSwitchPortMaskToGpu( gpuKey, connectedSwitches );

    // Reset and drain corresponding switch ports
    std::map <SwitchKeyType, uint64_t>::iterator it;
    for ( it = connectedSwitches.begin(); it != connectedSwitches.end(); it++ ) {
        SwitchKeyType switchKey = it->first;
        uint64_t switchPortMask = it->second;

        FM_LOG_DEBUG("lwLinkSendResetSwitchLinks switch %d switchPortMask 0x%lx",
                     switchKey.physicalId, switchPortMask);

        if ( GFMHelper::lwLinkSendResetSwitchLinks(nodeId, switchKey.physicalId,
                                                   switchPortMask, mpGfm->mLinkTrainIntf) ) {
            FM_LOG_WARNING("failed to reset LWSwitch access LWLinks for GPU uuid %s", gpuUuid);
            // best effort, just log the error and continue.
        }
    }

    return FM_INT_ST_OK;
}

/* 
Following are the cases where degraded Mode path intersects with the GPU reset path
and the corresponding actions done

Case 1) Reset Of Degraded GPU
Action- No need to train its LWLinks. (As we are supposed to Turn Off All Links)

Case 2) Reset When a Switch is Degraded
Action- Need to turn off at least two LWLinks -- Degraded Switch is removed, 
        so those Access Links wouldnt be trained to high anyway. So no specific action required

MIG Cases
Case 1) GPU was in MIG mode when FM started
     i. Access Links fails during re-training
Action- We do not do any action. Degraded mode runs only during FM initialization and not during runtime.
        This will result in reset failing.
    ii. All Access Links Trained
Action- If there was some switch degraded, ensure we should train off 2 LWLinks after mig is disabled.
*/ 

//
// process fm internal API command FM_COMPLETE_GPU_RESET
// Step 4 from LWML for GPU reset process. 
// FM will do the following
//   Attach the GPU
//   Train LWLinks to high speed
//   Configure the GPU fabric address
//
FMIntReturn_t
GlobalFmApiHandler::completeGpuReset(char *gpuUuid)
{
    FMIntReturn_t rc;

    rc = isGpuResetSupported(gpuUuid);
    if (rc != FM_INT_ST_OK) {
        return rc;
    }

    FM_LOG_INFO("processing request to complete GPU uuid %s reset", gpuUuid);

    rc = completeGpuResetHelper(gpuUuid);

    // remove gpu from ResetGPUList
    if (mpGfm->getFabricMode() == FM_MODE_VGPU) {
        mpGfm->mGfmPartitionMgr->remGpuFromResetPendingList(gpuUuid);
    }

    return rc;
} 

FMIntReturn_t
GlobalFmApiHandler::completeGpuResetHelper(char *gpuUuid)
{
    FMIntReturn_t rc;
    uint32_t nodeId = 0; // API does not have nodeId, use 0 now.
    uint64_t gpuEnabledLinkMask;

    // serialize all FM APIs
    FMAutoLock lock(mLock);

    // Attach to the GPU
    rc = mpGfm->mpConfig->configAttachGpu(nodeId, gpuUuid);
    if ( rc != FM_INT_ST_OK ) {
        // failed to send the attach request
        return rc;
    }

    //
    // refresh GPU info, as GPU can go into or out of MIG mode after reset and we need to fetch
    // new lwlink mask
    // Note: We can't blindly fetch and update all our GPU information map as when a group (or all) of
    // GPU's reset is issued, LWML go step by step for each GPU. Later when this final step comes, all those
    // GPUs are unbinded from Driver and we will end-up with only this (or GPU which are completed final step)
    // GPU in our context. So, update this specific GPU lwlink mask.
    //
    mpGfm->refreshGpuLWLinkMaskInfo(nodeId, gpuUuid);

    FMGpuInfo_t gpuInfo;
    if (mpGfm->getGpuInfo(gpuUuid, gpuInfo) == false) {
        FM_LOG_ERROR("specified GPU uuid %s is not found in fabric manager context after reattaching", gpuUuid);
        return FM_INT_ST_ILWALID_GPU;
    }

    // get LWLink device information from LWLinkCoereLib driver.
    GlobalFMLWLinkDevRepo lwLinkDevRepo;
    GFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mpGfm->mDevInfoMsgHndlr, lwLinkDevRepo);

    // do the lookup on devices returned LWLinkCoreLib driver.
    FMLWLinkDevInfo lwLinkDevInfo;
    if (lwLinkDevRepo.getDeviceInfo(nodeId, gpuInfo.pciInfo, lwLinkDevInfo) == false ) {
        //
        // this could happen if we are in the process of enabling MIG and GPUs don't have any LWLinks
        //
        FM_LOG_INFO("not re-training GPU access LWLinks as LWLinks are disabled for GPU uuid %s", gpuUuid);
        return FM_INT_ST_OK;
    }

    //
    // Note: We can't blindly fetch and update LWLink device information (mLWLinkDevRepo) here
    // as when a group of (or all) GPU's reset is issued, LWML go step by step and at this stage we have
    // only those GPUs which are attached is present in CoreLib. Instead we should be updating the 
    // required fields or devices only (like GPU mask update above.)
    //
    // Also, this update is required as if a GPU LWLinks are disabled when FM is started (say due to MIG),
    // then that device will not be present in mLWLinkDevRepo. When MIG is disabled and a corresponding
    // GPU reset comes, we train back the links. But we need to update mLWLinkDevRepo with this new
    // device otherwise, we will not be able to map that GPUs physical ID
    //
    // Also, when a GPU have MIG enabled, it will be removed from LWLinkCoreLib driver. But here we are not
    // removing that device from Global FM mLWLinkDevRepo map.
    //
    mpGfm->mLWLinkDevRepo.mergeDeviceInfoForNode(nodeId, lwLinkDevRepo);

	lwswitch::GpuDegradedReason reasonNotUsed;
    // if GPU is degraded, we can skip the link training as degraded GPU will be used only 
    // for non-lwlink workloads
    if (mpGfm->mpDegradedModeMgr->isGpuDegraded(gpuUuid)) {
        FM_LOG_INFO("not re-training GPU access LWLinks as GPU uuid %s is marked as degraded", gpuUuid);
        return FM_INT_ST_OK;
    }

    // we need to train the links. do LWLink initialization for the node
    GFMHelper::lwLinkInitializeNode(nodeId, mpGfm->mLinkTrainIntf, mpGfm->mLWLinkDevRepo);

    // discover all the intra node connections and construct a local connRepo
    GlobalFMLWLinkConnRepo lwLinkConnRepo;
    GFMHelper::lwLinkDiscoverIntraNodeConnOnNode(nodeId, mpGfm->mLinkTrainIntf, lwLinkConnRepo);

    uint64_t gpuLWLinkDrverId = lwLinkDevInfo.getDeviceId();

    // construct new connRepo that only has connections to the requested GPU
    GlobalFMLWLinkConnRepo linkConnRepoByDriverId;
    GFMHelper::getLWLinkConnsRepoByLWLinkDriverId(nodeId, gpuLWLinkDrverId, lwLinkConnRepo,linkConnRepoByDriverId);

    // train all connections to the GPU with the specific lwLinkGpuId
    int ret;
    if ( mpGfm->isParallelTrainingEnabled() ) {
        ret = GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(mpGfm, mpGfm->mLinkTrainIntf,
                                                                 linkConnRepoByDriverId,
                                                                 mpGfm->mLWLinkDevRepo,
                                                                 LWLINK_TRAIN_SAFE_TO_HIGH);
    } else {
        ret = GFMHelper::lwLinkTrainIntraNodeConnections(mpGfm, mpGfm->mLinkTrainIntf,
                                                         linkConnRepoByDriverId,
                                                         mpGfm->mLWLinkDevRepo,
                                                         LWLINK_TRAIN_SAFE_TO_HIGH);
    }

    if ( ret != 0 ) {
        FM_LOG_ERROR("failed to re-train GPU access LWLinks for GPU uuid %s", gpuUuid);
        return FM_INT_ST_GENERIC_ERROR;
    }

    int numDegradedLinks = 0;
    if (mpGfm->mpDegradedModeMgr->isAnyDeviceDegraded()) {
        // we get the number of degraded links for this GPU due to the check
        // below that tests for the number of acccess links. This function just
        // computes the number of degraded switch pairs, and then multiplies 
        // that value by 2 (as GPU reset is supported only in Ampere). 
        // This was done, instead of updating the mGpuInfoMap because of cases 
        // where MIG might be enabled for a GPU. In such a case we will not have the physical ID
        // of the GPU. Since the gpulinkmask is computed with the help of the mpFabric, which only has info
        // about the physical ID and not uuid, such an update to mGpuInfoMap isnt possible.
        // NOTE: Should revisit for future architectures, and multi node
        numDegradedLinks = mpGfm->mpDegradedModeMgr->getNumDegradedLinksForGpu();
    }

    // check whether all the access connections are detected and trained to high speed
    uint32 detectedAccessConnCnt = mpGfm->mTopoValidator->getAccessConnCount(nodeId, linkConnRepoByDriverId);
    uint32 enabledLinkCount = getGpuNumEnabledLinks(gpuInfo);

    if (detectedAccessConnCnt != (enabledLinkCount - numDegradedLinks)) {
        FM_LOG_ERROR("not all the LWLink access connections are detected for GPU uuid %s", gpuUuid);
        //
        // nothing much FM can do now to recover. may be whole system reset/reboot is required to recover.
        // also, LWML/lwpu-smi will print similar warning for user
        //
        return FM_INT_ST_GENERIC_ERROR;        
    }

    // verify all the GPU access LWLinks are trained and active
    if (!mpGfm->mTopoValidator->isAccessConnsActive(nodeId, linkConnRepoByDriverId, mpGfm->mLWLinkDevRepo)) {
        FM_LOG_ERROR("not all the LWLink access connections are trained to ACTIVE for GPU uuid %s", gpuUuid);
        //
        // nothing much FM can do now to recover. may be whole system reset/reboot is required to recover.
        // also, LWML/lwpu-smi will print similar warning for user
        //
        return FM_INT_ST_GENERIC_ERROR;        
    }

    // The link training is done on a local lwLinkConnRepo
    // update connRepo maintained by GFM i.e.mpGfm->mLWLinkConnRepo
    updateLWLinkConnsRepo(nodeId, linkConnRepoByDriverId, mpGfm->mLWLinkConnRepo);

    // refresh GPU connection mapping upon successful training
    mpGfm->mTopoValidator->mapGpuIndexByLWLinkConns(mpGfm->mGpuLWLinkConnMatrix);

    uint32_t physicalId;
    if (mpGfm->getGpuPhysicalId(nodeId, gpuUuid, physicalId) == false) {
        // still not able to find GPU mapping after refreshing the connection matrix
        FM_LOG_ERROR("failed to find LWLink mapping for GPU uuid %s", gpuUuid);
        return FM_INT_ST_GENERIC_ERROR;
    }

    GpuKeyType gpuKey;
    gpuKey.nodeId = nodeId;
    gpuKey.physicalId = physicalId;

    // Re configure all switch ports that are connected to this GPU
    // as switch port reset wipe off all port config and routing table entries on the port
    rc = reconfigSwitchPortAfterGpuReset(gpuKey);
    if ( rc != FM_INT_ST_OK ) {
        // failed to send the config request
        return rc;
    }

    // Configure the GPU fabric address
    rc = mpGfm->mpConfig->configGpu(nodeId, gpuUuid);
    if ( rc != FM_INT_ST_OK ) {
        // failed to send the config request
        return rc;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFmApiHandler::reconfigSwitchPortAfterGpuReset(GpuKeyType gpuKey)
{
    FMIntReturn_t rc;

    std::map <SwitchKeyType, uint64_t> connectedSwitches;
    mpGfm->mpParser->getSwitchPortMaskToGpu( gpuKey, connectedSwitches );

    std::map <SwitchKeyType, uint64_t>::iterator it;
    for ( it = connectedSwitches.begin(); it != connectedSwitches.end(); it++ )
    {
        SwitchKeyType switchKey = it->first;
        uint64_t switchPortMask = it->second;

        PortKeyType portKey;
        portKey.nodeId = switchKey.nodeId;
        portKey.physicalId = switchKey.physicalId;
        lwswitch::SwitchDegradedReason reasonNotUsed;

        // do not configure degraded switch
        if (mpGfm->mpDegradedModeMgr->isSwitchDegraded(switchKey.nodeId, switchKey.physicalId, reasonNotUsed)) {
            FM_LOG_INFO("not configuring LWSwitch physical id %d after GPU reset as it is marked as degraded", switchKey.physicalId);
            continue;
        }

        // find all switch ports that are connected to this GPU
        std::list<PortKeyType> portList;
        portList.clear();

        for (uint32_t portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( ( switchPortMask & ( (uint64_t)1 << portIndex ) ) != 0 )
            {
                portKey.portIndex = portIndex;
                portList.push_back( portKey );
            }
        }

        // reconfigure the port and routing tables on the ports, as all port config
        // are gone after switch port reset
        rc = mpGfm->mpConfig->configSwitchPortList(gpuKey.nodeId, ILWALID_FABRIC_PARTITION_ID,
                                                   portList, true);
        if ( rc != FM_INT_ST_OK ) {
            // failed to switch port request
            return rc;
        }

        std::list<PortKeyType>::iterator portIt;
        for (portIt = portList.begin(); portIt != portList.end(); portIt++) {
            PortKeyType portKey = *portIt;

            SwitchKeyType switchKey;
            switchKey.nodeId = portKey.nodeId;
            switchKey.physicalId = portKey.physicalId;

            std::map <SwitchKeyType, lwswitch::switchInfo * >::iterator switchIt;
            switchIt = mpGfm->mpParser->lwswitchCfg.find( switchKey );
            if ( switchIt == mpGfm->mpParser->lwswitchCfg.end() ) {
                FM_LOG_ERROR("failed to find LWSwitch for physical id %d", portKey.physicalId);
                return FM_INT_ST_GENERIC_ERROR;
            }

            lwswitch::switchInfo *switchInfo = switchIt->second;
            rc = mpGfm->mpConfig->configRmapEntries(portKey.nodeId, switchInfo, portKey.portIndex, true);
            if ( rc != FM_INT_ST_OK ) {
                // failed to config rmap table
                return rc;
            }

            rc = mpGfm->mpConfig->configRidEntries(portKey.nodeId, switchInfo, portKey.portIndex, true);
            if ( rc != FM_INT_ST_OK ) {
                // failed to config rid table
                return rc;
            }

            rc = mpGfm->mpConfig->configRlanEntries(portKey.nodeId, switchInfo, portKey.portIndex, true);
            if ( rc != FM_INT_ST_OK ) {
                // failed to config rlan table
                return rc;
            }
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFmApiHandler::isGpuResetSupported(char *gpuUuid)
{
    uint32_t nodeId = 0; // API does not have nodeId, use 0 now.

    if (!gpuUuid) {
        FM_LOG_ERROR("GPU reset request with null GPU uuid parameter");
        return FM_INT_ST_ILWALID_GPU;
    }

    FMGpuInfo_t gpuInfo;
    if (mpGfm->getGpuInfo(gpuUuid, gpuInfo) == false) {
        FM_LOG_ERROR("specified GPU uuid %s is not found or attached by fabric manager", gpuUuid);
        return FM_INT_ST_ILWALID_GPU;
    }

    if (gpuInfo.archType < LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100) {
        FM_LOG_ERROR("GPU reset request is not supported on GPU uuid %s", gpuUuid);
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // This functionality is not supported in Shared LWSwitch Mode.
    if (mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) {
        FM_LOG_ERROR("GPU reset is not supported in Shared LWSwitch multitenancy mode");
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // This functionality is not supported in vGPU Mode if given GPU is part of active partition.
    if ((mpGfm->getFabricMode() == FM_MODE_VGPU) && (mpGfm->mGfmPartitionMgr->isGpuInActivePartition(nodeId, gpuUuid))) {
        FM_LOG_ERROR("Cannot reset specified GPU uuid %s as it is part of an active vGPU partition", gpuUuid);
        return FM_INT_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mpGfm->isInitDone() == false) {
        return FM_INT_ST_NOT_CONFIGURED;
    }

    return FM_INT_ST_OK;
}

// update the destLinkConnRepo with the connections in srcLinkConnRepo
void
GlobalFmApiHandler::updateLWLinkConnsRepo(uint32_t nodeId,
                                          GlobalFMLWLinkConnRepo &srcLinkConnRepo,
                                          GlobalFMLWLinkConnRepo &destLinkConnRepo)
{
    LWLinkIntraConnMap &srcIntraConnMap = srcLinkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = srcIntraConnMap.find(nodeId);
    if (it == srcIntraConnMap.end()) {
        // no conn in srcLinkConnRepo for this nodeId
        return;
    }

    FMLWLinkDetailedConnInfoList &srcIntraConnList = it->second;
    destLinkConnRepo.mergeIntraConnections(nodeId, srcIntraConnList);
}

uint32_t
GlobalFmApiHandler::getGpuNumEnabledLinks(FMGpuInfo_t &gpuInfo)
{
    uint32_t enabledLinkCount = 0;
    uint64_t enabledLinkMask = gpuInfo.enabledLinkMask;

    while (enabledLinkMask) {
        enabledLinkCount += enabledLinkMask & 1;
        enabledLinkMask >>= 1;
    }

    return enabledLinkCount;
}

