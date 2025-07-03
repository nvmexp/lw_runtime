/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <stdexcept>

#include "lwRmApi.h"
#include "lwos.h"
#include "ctrl0000.h"
#include "ctrl0000gpu.h"
#include "class/cl000f.h"
#include "ctrl/ctrl000f.h"

//#include "fm_log.h"
#include "LocalFMGpuMgr.h"
//#define printf(...) {}

#define FM_SESSION_HANDLE 0x80000000
#define FM_GPU_DEVICE_HANDLE_BASE (FM_SESSION_HANDLE + 1)
const LwU32 LocalFMGpuMgr::mGpuDevHandleBase = FM_GPU_DEVICE_HANDLE_BASE;
const LwU32 LocalFMGpuMgr::mGpuSubDevHandleBase = FM_GPU_DEVICE_HANDLE_BASE + LW0000_CTRL_GPU_MAX_PROBED_GPUS;

LocalFMGpuMgr::LocalFMGpuMgr()
{
    printf( "entering LocalFMGpuMgr constructor\n");
    LwU32 rmResult;

    mFMSessionFd = 0;
    mInitDoneGpuInfoMap.clear();

    // allocate our rmclient object
    rmResult = LwRmAllocRoot(&mRmClientFd);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        printf( "failed to allocate handle (client) to LWPU GPU Driver with error:%s\n",
                        lwstatusToString(rmResult));
        throw std::runtime_error("failed to allocate handle (client) to LWPU GPU Driver");
    }
}

LocalFMGpuMgr::~LocalFMGpuMgr()
{
    printf( "entering LocalFMGpuMgr destructor\n");

    // free fm session
    if (0 != mFMSessionFd) {
        freeFmSession();
        mFMSessionFd = 0;
    }

    // detach and close all GPU handles
    if (0 != mInitDoneGpuInfoMap.size()) {
        deInitializeAllGpus();
    }

    mInitDoneGpuInfoMap.clear();

    // free our FM client fd at the end.
    LwRmFree(mRmClientFd, LW01_NULL_OBJECT, mRmClientFd);
    mRmClientFd = 0;
}

/******************************************************************************************
 Method to attach(initialize) all GPUs. Used by LocalFM during initialization time
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::initializeAllGpus()
{
    std::vector<LwU32> gpuIdList;
    std::vector<LwU32>::iterator it;
    FMIntReturn_t fmResult;

    printf( "entering initializeAllGpus\n");
    //
    // bail out if GPUs are already attached. This is supposed to be
    // called during initialization time
    //
    if (0 != mInitDoneGpuInfoMap.size()) {
        printf( "request to initialize all GPUs when some GPUs are already initialized\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    // get list of gpus and its IDs from RM
    getProbedGpuCount(gpuIdList);

    // attach each probed GPUs  
    for (it = gpuIdList.begin(); it != gpuIdList.end(); it++) {
        LwU32 gpuId = (*it);

        printf( "Initializing GPU ID = %d\n", gpuId);

        // get the pci bdf information
        FMPciInfo_t pciInfo;
        fmResult = getGpuPciInfo(gpuId, pciInfo);
        if (FM_INT_ST_OK != fmResult) {
            // error already logged
            return fmResult;
        }

        // get gpu uuid information
        FMUuid_t uuid;
        fmResult = getGpuUuidInfo(gpuId, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // UUID query may not succeed for all GPUs (like consumer GPUs/old VBIOS), skip them
            printf( "unable to get UUID, skipping GPU BDF: %s ID: %d\n",
                           pciInfo.busId, gpuId);
            continue;
        }

        // attach and open all required GPU handles
        fmResult = attachAndOpenGpuHandles(gpuId, pciInfo, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // error already logged and clean-up done.
            return fmResult;
        }
    }

    return FM_INT_ST_OK;
}

/******************************************************************************************
 Method to detach(deinitialize) all GPUs. Used by LocalFM during shutdown time or after
 shared LWSwitch model initialization
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::deInitializeAllGpus()
{
    printf( "entering deInitializeAllGpus");

    if (0 == mInitDoneGpuInfoMap.size()) {
        printf( "request to deinitialize all GPUs when no GPUs are initialized\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    InitDoneGpuInfoMap_t::iterator it;
    for (it = mInitDoneGpuInfoMap.begin(); it != mInitDoneGpuInfoMap.end(); it++ ) {
        FMUuid_t uuid = it->first;
        GpuCtxInfo_t &gpuCtxInfo = it->second;

        // close all the device and sub device handles
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
    }

    // all the GPUs are deinitialized. clear our context.
    mInitDoneGpuInfoMap.clear();
    return FM_INT_ST_OK;    
}

/******************************************************************************************
 Method to attach(initialize) a single GPU. Used by LocalFM during partition activation 
 in shared LWSwitch model 
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::initializeGpu(FMUuid_t &uuidToInit)
{
    printf( "entering initializeGpu\n");

    FMIntReturn_t fmResult;

    // check whether the GPU is already attached and initialized.
    InitDoneGpuInfoMap_t::iterator initDoneIt = mInitDoneGpuInfoMap.find(uuidToInit);
    if (initDoneIt != mInitDoneGpuInfoMap.end()) {
        printf("trying to initialize an already initialized GPU, uuid: %s\n", uuidToInit.bytes);
        return FM_INT_ST_BADPARAM;
    }

    // query RM to get a list of lwrrently probed GPUs.
    std::vector<LwU32> gpuIdList;
    std::vector<LwU32>::iterator it;
    getProbedGpuCount(gpuIdList);

    // attach each probed GPUs  
    for (it = gpuIdList.begin(); it != gpuIdList.end(); it++) {
        LwU32 gpuId = (*it);

        // get the pci bdf information
        FMPciInfo_t pciInfo;
        fmResult = getGpuPciInfo(gpuId, pciInfo);
        if (FM_INT_ST_OK != fmResult) {
            // error already logged
            return fmResult;
        }

        // get gpu uuid information
        FMUuid_t uuid;
        fmResult = getGpuUuidInfo(gpuId, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // UUID query may not succeed for all GPUs (like consumer GPUs/old VBIOS), skip them
            printf( "unable to get UUID, skipping GPU BDF: %s ID: %d\n",
                           pciInfo.busId, gpuId);
            continue;
        }

        // attach if the uuid matches
        if (uuidToInit == uuid) {
            // attach and open all required GPU handles
            // error already logged and clean-up done.
            fmResult = attachAndOpenGpuHandles(gpuId, pciInfo, uuid);
            return fmResult;
        }
    }

    // not found the required GPU/uuid
    return FM_INT_ST_BADPARAM;
}

/******************************************************************************************
 Method to detach(deinitialize) a single GPU. Used by LocalFM during partition deactivation 
 in shared LWSwitch model 
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::deInitializeGpu(FMUuid_t &uuidToDeinit)
{
    printf( "entering deInitializeGpu");

    FMIntReturn_t fmResult;

    // check whether the GPU is already attached and initialized.
    InitDoneGpuInfoMap_t::iterator initDoneIt = mInitDoneGpuInfoMap.find(uuidToDeinit);
    if (initDoneIt == mInitDoneGpuInfoMap.end()) {
        printf("trying to deinitialize a GPU which is not initialized, uuid: %s\n", uuidToDeinit.bytes);
        return FM_INT_ST_BADPARAM;
    }

    FMUuid_t uuid = initDoneIt->first;
    GpuCtxInfo_t &gpuCtxInfo = initDoneIt->second;
    // close all the device and sub device handles
    detachAndCloseGpuHandles(gpuCtxInfo, uuid);

    // remove the specified GPU from init done map
    mInitDoneGpuInfoMap.erase(initDoneIt);

    return FM_INT_ST_OK;
}


FMIntReturn_t
LocalFMGpuMgr::getInitializedGpuInfo(std::vector<FMGpuInfo_t> &initDoneGpuInfo)
{
    initDoneGpuInfo.clear();
    InitDoneGpuInfoMap_t::iterator it;
    for (it = mInitDoneGpuInfoMap.begin(); it != mInitDoneGpuInfoMap.end(); it++ ) {
        FMUuid_t uuid = it->first;
        GpuCtxInfo_t &gpuCtxInfo = it->second;
        FMGpuInfo_t gpuInfo;
        gpuInfo.gpuIndex = gpuCtxInfo.deviceInstance;
        gpuInfo.uuid = uuid;
        gpuInfo.pciInfo = gpuCtxInfo.pciInfo;
        initDoneGpuInfo.push_back(gpuInfo);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getBlacklistGpuInfo(std::vector<FMBlacklistGpuInfo_t> &blacklistGpuInfo)
{
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probeParams = {{0}};
    LwU32 rmResult;

    blacklistGpuInfo.clear();

    // get GPUs detected/probed by RM
    rmResult = LwRmControl(mRmClientFd, mRmClientFd,
                           LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probeParams, sizeof(probeParams));

    if (LWOS_STATUS_SUCCESS != rmResult) {       
        printf( "request to get list of probed GPUs from LWPU GPU Driver failed with error: %s\n",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    for (int idx = 0; idx < LW0000_CTRL_GPU_MAX_PROBED_GPUS; idx++) {
        if (probeParams.excludedGpuIds[idx] != LW0000_CTRL_GPU_ILWALID_ID) {
            LwU32 gpuId = probeParams.excludedGpuIds[idx];
            FMBlacklistGpuInfo_t gpuInfo;
            FMIntReturn_t fmResult;
            // get pci BDI information
            fmResult = getGpuPciInfo(gpuId, gpuInfo.pciInfo);
            if (FM_INT_ST_OK != fmResult) {
                // error already logged
                return fmResult;
            }
            // get gpu uuid information
            fmResult = getGpuUuidInfo(gpuId, gpuInfo.uuid);
            if (FM_INT_ST_OK != fmResult) {
                // UUID query for blacklist GPUs should succeed as RM uses UUID to blacklist.
                // error already logged
                return fmResult;
            }
            // got all the information for this GPU
            blacklistGpuInfo.push_back(gpuInfo);
        }
    }

    return FM_INT_ST_OK;
}


FMIntReturn_t
LocalFMGpuMgr::allocFmSession()
{
    LwU32 rmResult;
    LW000F_ALLOCATION_PARAMETERS fmSessionAllocParams = {0};

    if (0 != mFMSessionFd) {
        printf( "Fabric Manager session for LWPU GPU driver is already allocated\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmAlloc(mRmClientFd, mRmClientFd, FM_SESSION_HANDLE,
                         FABRIC_MANAGER_SESSION, &fmSessionAllocParams);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to allocate Fabric Manager session for LWPU GPU driver, error: %s\n",
                      lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    mFMSessionFd = FM_SESSION_HANDLE;
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::freeFmSession()
{
    if (0 == mFMSessionFd) {
        printf( "Fabric Manager session for LWPU GPU driver is not allocated\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    LwRmFree(mRmClientFd, mRmClientFd, mFMSessionFd);
    mFMSessionFd = 0;

    return FM_INT_ST_OK;
}

FMIntReturn_t 
LocalFMGpuMgr::fmSessionSetState()
{
    LwU32 rmResult;

    if (0 == mFMSessionFd) {
        printf( "Fabric Manager session for LWPU GPU driver is not allocated\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmControl(mRmClientFd, mFMSessionFd, LW000F_CTRL_CMD_SET_FM_STATE, NULL, 0);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to set Fabric Manager session in LWPU GPU driver, error: %s\n",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::fmSessionClearState()
{
    LwU32 rmResult;

    if (0 == mFMSessionFd) {
        printf( "Fabric Manager session for LWPU GPU driver is not allocated\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmControl(mRmClientFd, mFMSessionFd, LW000F_CTRL_CMD_CLEAR_FM_STATE, NULL, 0);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to clear Fabric Manager session in LWPU GPU driver, error: %s\n",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::setGpuFabricBaseAddress(FMUuid_t gpuUuid, unsigned long long baseAddress)
{
    LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS addressParams = {0};
    LwU32 rmResult;

    // check whether the GPU is nitialized before setting base address
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        printf("trying to set fabric base address for an uninitialized GPU, uuid: %s\n", gpuUuid.bytes);
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;

    addressParams.fabricBaseAddr = baseAddress;
    rmResult = LwRmControl(mRmClientFd, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR, &addressParams, sizeof(addressParams));

    if (LWOS_STATUS_SUCCESS != rmResult) {       
        printf( "request to set fabric base address for GPU uuid: %s failed with error: %s\n",
                     gpuUuid.bytes, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getProbedGpuCount(std::vector<LwU32> &gpuIds)
{
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probeParams = {{0}};
    LwU32 rmResult;

    gpuIds.clear();

    // get GPUs detected/probed by RM
    rmResult = LwRmControl(mRmClientFd, mRmClientFd,
                           LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probeParams, sizeof(probeParams));

    if (LWOS_STATUS_SUCCESS != rmResult) {       
        printf( "request to get list of probed GPUs from LWPU GPU Driver failed with error: %s\n",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    for (int idx = 0; idx < LW0000_CTRL_GPU_MAX_PROBED_GPUS; idx++) {
        if (probeParams.gpuIds[idx] != LW0000_CTRL_GPU_ILWALID_ID) {
            gpuIds.push_back(probeParams.gpuIds[idx]);
            printf("found one gpu\n");
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::attachAndOpenGpuHandles(LwU32 gpuId, FMPciInfo_t &pciInfo, FMUuid_t &uuid)
{
    FMIntReturn_t fmResult;

    GpuCtxInfo_t gpuCtxInfo = {0};
    gpuCtxInfo.gpuId = gpuId;
    gpuCtxInfo.pciInfo = pciInfo;

    printf( "attaching and opening GPU BDF: %s ID: %d\n", gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId);

    // skip drain state enabled GPUs
    if (isGpuInDrainState(gpuCtxInfo.gpuId)) {
        printf( "not attaching/opening drain state enabled GPU BDF: %s ID: %d\n",
                    gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId);
        return FM_INT_ST_OK;
    }
    printf( "GPU not in drain state\n");

    // attach the GPU
    fmResult = doRmAttachGpu(gpuCtxInfo.gpuId);
    if (FM_INT_ST_OK != fmResult) {
        printf( "unable to open/attach GPU handle for GPU BDF: %s ID: %d\n",
                     gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId);
        return fmResult;
    }

    // fetch device instance id
    fmResult = getGpuRmDevInstanceIdInfo(gpuCtxInfo);
    if (FM_INT_ST_OK != fmResult) {
        // error already logged, do the clean-up
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
        return fmResult;
    }

    // allocate all sub devices and handles
    fmResult = allocateGpuSubDevices(gpuCtxInfo);
    if (FM_INT_ST_OK != fmResult) {
        // error already logged, do the clean-up
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
        return fmResult;
    }

    // all looks good. keep/cache the information in our context
    mInitDoneGpuInfoMap.insert(std::make_pair(uuid, gpuCtxInfo));
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::detachAndCloseGpuHandles(GpuCtxInfo_t &gpuCtxInfo, FMUuid_t &uuid)
{
    printf( "detachAndCloseGpuHandles for GPU BDF: %s\n", gpuCtxInfo.pciInfo.busId);

    freeGpuSubDevices(gpuCtxInfo);

    doRmDeatchGpu(gpuCtxInfo.gpuId);

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuPciInfo(LwU32 gpuId, FMPciInfo_t &pciInfo)
{
    LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS pciInfoParams = {0};
    LwU32 rmResult;

    printf( "Getting PCI info for GPU ID = %d\n", gpuId);

    pciInfoParams.gpuId = gpuId;

    rmResult = LwRmControl(mRmClientFd, mRmClientFd,
                           LW0000_CTRL_CMD_GPU_GET_PCI_INFO, &pciInfoParams, sizeof(pciInfoParams));

    if (LWOS_STATUS_SUCCESS != rmResult) {       
        printf( "failed to get PCI information from LWPU GPU Driver for GPU id: %d, error: %s\n",
                     gpuId,  lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    pciInfo.domain = pciInfoParams.domain;
    pciInfo.bus = pciInfoParams.bus;
    pciInfo.device = pciInfoParams.slot;
    pciInfo.function = 0; //RM don't provide this information.
    snprintf(pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
              FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&pciInfo));

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuUuidInfo(LwU32 gpuId, FMUuid_t &uuidInfo)
{
    LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS uuidParams = {0};
    LwU32 rmResult;

    printf( "Getting UUID info for GPU ID = %d\n", gpuId);

    uuidParams.gpuId = gpuId;
    uuidParams.flags = LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_ASCII;

    rmResult = LwRmControl(mRmClientFd, mRmClientFd, LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID,
                           &uuidParams, sizeof(uuidParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {       
        printf( "failed to get UUID information from LWPU GPU Driver for GPU id: %d, error:%s\n",
                     gpuId,  lwstatusToString(rmResult));

        //return FM_INT_ST_GENERIC_ERROR;
        memset(uuidInfo.bytes, gpuId + '0', sizeof(uuidInfo.bytes));
        uuidInfo.bytes[sizeof(uuidInfo.bytes)] = '\0' ;
        printf( "Fake GPU ID = %d UUID = %s\n", gpuId, uuidInfo.bytes);
        return FM_INT_ST_OK;
        //TODO
        return FM_INT_ST_GENERIC_ERROR;
    }

    strncpy(uuidInfo.bytes, (char *)uuidParams.gpuUuid, FM_UUID_BUFFER_SIZE);

    printf( "GPU ID = %d UUID = %s\n", gpuId, uuidInfo.bytes);

    return FM_INT_ST_OK;
}

bool
LocalFMGpuMgr::isGpuInDrainState(LwU32 gpuId)
{
    LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS drainParms = {0};
    LwU32 rmResult;
    bool drainState;

    drainParms.gpuId = gpuId;
    rmResult = LwRmControl(mRmClientFd, mRmClientFd, LW0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE,
                           &drainParms, sizeof(drainParms));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        // drain state query failed, mark the GPU in drain state.
        return true;
    }

    if (LW0000_CTRL_GPU_DRAIN_STATE_ENABLED == drainParms.newState) {
        drainState = true;
    }

    return drainState;
}

FMIntReturn_t
LocalFMGpuMgr::doRmAttachGpu(LwU32 gpuId)
{
    LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams = {{0}};
    LwU32 rmResult;

    attachParams.gpuIds[0] = gpuId;
    attachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;

    rmResult = LwRmControl(mRmClientFd, mRmClientFd, LW0000_CTRL_CMD_GPU_ATTACH_IDS,
                           &attachParams, sizeof(attachParams));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to open/attach handle to GPU id: %d, error: %s\n",
                      gpuId,  lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // successfully attached gpu device
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::doRmDeatchGpu(LwU32 gpuId)
{
    LW0000_CTRL_GPU_DETACH_IDS_PARAMS detachParams = {{0}};
    LwU32 rmResult;

    detachParams.gpuIds[0] = gpuId;
    detachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;

    rmResult = LwRmControl(mRmClientFd, mRmClientFd, LW0000_CTRL_CMD_GPU_DETACH_IDS,
                           &detachParams, sizeof(detachParams));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to close/detach handle to GPU id: %d, error: %s\n",
                      gpuId,  lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // successfully deatched gpu device
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuRmDevInstanceIdInfo(GpuCtxInfo_t &gpuCtxInfo)
{
    LW0000_CTRL_GPU_GET_ID_INFO_PARAMS idInfoParams = {0};
    LwU32 rmResult;

    idInfoParams.gpuId = gpuCtxInfo.gpuId;
    rmResult = LwRmControl(mRmClientFd, mRmClientFd, LW0000_CTRL_CMD_GPU_GET_ID_INFO,
                           &idInfoParams, sizeof(idInfoParams));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        printf( "failed to get device instance id information for GPU id: %d, error: %s\n", 
                      idInfoParams.gpuId,  lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    gpuCtxInfo.deviceInstance = idInfoParams.deviceInstance;
    gpuCtxInfo.subDeviceInstance = idInfoParams.subDeviceInstance;
    gpuCtxInfo.gpuInstance = idInfoParams.gpuInstance;

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::allocateGpuSubDevices(GpuCtxInfo_t &gpuCtxInfo)
{
    LW0080_ALLOC_PARAMETERS allocDeviceParams = {0};
    LwU32 rmResult;

    allocDeviceParams.deviceId = gpuCtxInfo.deviceInstance;
    LwU32 lw0080handle = getDeviceHandleForGpu(gpuCtxInfo.gpuId);

    rmResult = LwRmAlloc(mRmClientFd, mRmClientFd, lw0080handle,
                         LW01_DEVICE_0, &allocDeviceParams);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        printf( "Failed to allocate GPU base device instance for GPU id: %d error: %s\n",
                     gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // cache the handle after succesful allocation
    gpuCtxInfo.lw0080handle = lw0080handle;

    // allocate sub devices
    LW2080_ALLOC_PARAMETERS allocSubDeviceParams = {0};
    allocSubDeviceParams.subDeviceId = 0;
    LwU32 lw2080handle = getMemSubDeviceHandleForGpu(gpuCtxInfo.gpuId);
    rmResult = LwRmAlloc(mRmClientFd, gpuCtxInfo.lw0080handle, lw2080handle,
                         LW20_SUBDEVICE_0, &allocSubDeviceParams);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        printf( "Failed to allocate GPU sub device instance for GPU id: %d error: %s\n",
                     gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // cache the handle after succesful allocation    
    gpuCtxInfo.lw2080handle = lw2080handle;

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::freeGpuSubDevices(GpuCtxInfo_t &gpuCtxInfo)
{
    if (gpuCtxInfo.lw2080handle) {
        LwRmFree(mRmClientFd, gpuCtxInfo.lw2080handle, gpuCtxInfo.lw2080handle);
        gpuCtxInfo.lw2080handle = 0;
    }

    if (gpuCtxInfo.lw0080handle) {
        LwRmFree(mRmClientFd, gpuCtxInfo.lw0080handle, gpuCtxInfo.lw0080handle);
        gpuCtxInfo.lw0080handle = 0;
    }

    return FM_INT_ST_OK;
}

LwU32
LocalFMGpuMgr::getDeviceHandleForGpu(LwU32 gpuIndex)
{
    return (mGpuDevHandleBase + gpuIndex);
}

LwU32
LocalFMGpuMgr::getMemSubDeviceHandleForGpu(LwU32 gpuIndex)
{
    return (mGpuSubDevHandleBase + gpuIndex);
}
