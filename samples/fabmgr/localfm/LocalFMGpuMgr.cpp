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
#include <stdexcept>
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "lwRmApi.h"
#include "lwos.h"
#include "ctrl0000.h"
#include "ctrl0000gpu.h"
#include "class/cl000f.h"
#include "class/cl0000.h"
#include "class/cl0005.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl000f.h"
#include "ctrl/ctrl000f_imex.h"
#include "ctrl2080event.h"
#include "ctrl2080fla.h"
#include "ctrl2080mc.h"
#include "ctrl2080lwlink.h"
#include "ctrl/ctrl00f4.h"
#include "ctrl/ctrl00f5.h"

#include "fm_log.h"
#include "LocalFMGpuMgr.h"
#include "FMAutoLock.h"
#include "FMHandleGenerator.h"

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#include "modsdrv.h"

#define LwRmAllocOsEvent ModsAllocOsEvent
#define LwRmFreeOsEvent  ModsFreeOsEvent
#define LwRmGetEventData ModsGetEventData

namespace
{
    std::map<LwU32,void*> s_ModsEventMap;
    LwU32 ModsAllocOsEvent(LwHandle hClient, LwHandle hDevice, LwHandle *hOsEvent, void *fd)
    {
        LwU32* fdId = (LwU32*)fd;
        void* modsEvent = ModsDrvAllocEvent("LocalFmGpuMgr");
        void* osEvent = ModsDrvGetOsEvent(modsEvent, hClient, hDevice);
        *fdId = *((LwU32*)osEvent);
        s_ModsEventMap[*fdId] = modsEvent;
        return LW_OK;
    }
    LwU32 ModsFreeOsEvent(LwHandle hClient, LwHandle hDevice, LwU32 fd)
    {
        if (s_ModsEventMap.find(fd) == s_ModsEventMap.end())
        {
            FM_LOG_WARNING("Can't find fd %d in event map", fd);
            return LW_ERR_ILWALID_PARAMETER;
        }
        void* modsEvent = s_ModsEventMap[fd];
        ModsDrvFreeEvent(modsEvent);
        s_ModsEventMap.erase(fd);
        return LW_OK;
    }
    LwU32 ModsGetEventData(LwHandle hClient, LwU32 fd, void *pEventData, LwU32 *pMoreEvents)
    {
        // TODO
        *pMoreEvents = 0;
        return LW_OK;
    }
};
#endif

LocalFMGpuMgr::LocalFMGpuMgr()
{
    FM_LOG_DEBUG("entering LocalFMGpuMgr constructor");
    LwU32 rmResult;
    mFMSessionHandle = 0;
    mInitDoneGpuInfoMap.clear();
    mFdToGpuUuidMap.clear();
    mEventDescriptorList.clear();
    mEventSubscriberInfoMap.clear();
    mFabricEventsFd = -1;
    mFabricEventsHandle = 0;

    lwosInitializeCriticalSection( &mLock );

    // allocate our rmclient object
    rmResult = LwRmAllocRoot(&mRmClientHandle);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_CRITICAL("failed to allocate handle (client) to LWPU GPU driver with error:%s",
                        lwstatusToString(rmResult));
        // throw error based on common return values
        switch (rmResult) {
            case LW_ERR_OPERATING_SYSTEM: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver. "
                                         "Make sure that the LWPU driver is installed and running");
                break;
            }
            case LW_ERR_INSUFFICIENT_PERMISSIONS: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver. "
                                         "Make sure that the current user has permission to access device files");
                break;
            }
            default: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver. "
                                         "Check Fabric Manager log for detailed error information");
                break;
            }
        } // end of switch
    }

    //
    // create a socketpair to break from gpu event polling loop. socketpair has two fds and mEventLoopBreakFds[0]
    // used for watching in poll() and mEventLoopBreakFds[1] will be used to write which will signal the poll()
    // When a GPU is de-initialized, (like during GPU reset), we will close all the GPU handles. But the gpu
    // event polling (see watchEvents) thread is still blocked on poll() with those fds, leaving outstanding handles.
    // this socketpair will be written so that the poll() will unblock immediately instead of its 10 second timeout.
    //
    int retVal;
    retVal = socketpair(AF_UNIX, SOCK_SEQPACKET, SOCK_STREAM, mEventLoopBreakFds);
    if (retVal == -1) {
        std::ostringstream ss;
        ss << "request to create socketpair for GPU event loop signaling failed with error " << strerror(errno);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // make an attempt to set some desired flags for our socketpair fds
    int fdFlags;
    fdFlags = fcntl(mEventLoopBreakFds[0], F_GETFD, NULL);
    retVal = fcntl(mEventLoopBreakFds[0], F_SETFD, fdFlags | FD_CLOEXEC);

    if (retVal == -1) {
        std::ostringstream ss;
        ss << "request to set file descriptor flags for GPU event loop signaling failed with error " << strerror(errno);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    fdFlags = fcntl(mEventLoopBreakFds[1], F_GETFD, NULL);
    retVal = fcntl(mEventLoopBreakFds[1], F_SETFD, fdFlags | FD_CLOEXEC);

    if (retVal == -1) {
        std::ostringstream ss;
        ss << "request to set file descriptor flags for GPU event loop signaling failed with error " << strerror(errno);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // alloc event handle for fabric events (only one time)
    if (!FMHandleGenerator::allocHandle(mFabricEventsHandle)) {
        std::ostringstream ss;
        ss << "allocating fabric events handle failed";
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // initialize our gpu event polling loop exit signaling constructs
    lwosInitializeCriticalSection(&m_EventLoopBreakMutex);
    lwosCondCreate(&m_EventLoopBreakCondition);
}

LocalFMGpuMgr::~LocalFMGpuMgr()
{
    FM_LOG_DEBUG("entering LocalFMGpuMgr destructor");

    stopGpuEventWatch();

    if (0 != mFabricEventsHandle) {
        FMHandleGenerator::freeHandle(mFabricEventsHandle);
        mFabricEventsHandle = 0;
    }

    // free fm session
    if (0 != mFMSessionHandle) {
        freeFmSession();
        FMHandleGenerator::freeHandle(mFMSessionHandle);
        mFMSessionHandle = 0;
    }

    // detach and close all GPU handles
    if (0 != mInitDoneGpuInfoMap.size()) {
        deInitializeAllGpus();
    }
    mInitDoneGpuInfoMap.clear();
    // free our FM client handle at the end.
    LwRmFree(mRmClientHandle, LW01_NULL_OBJECT, mRmClientHandle);

    lwosDeleteCriticalSection( &mLock );
    mRmClientHandle = 0;

    // close our socketpair file descriptors
    close(mEventLoopBreakFds[0]);
    close(mEventLoopBreakFds[1]);
    lwosDeleteCriticalSection(&m_EventLoopBreakMutex);
    lwosCondDestroy(&m_EventLoopBreakCondition);

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
    // hold lock as this will modify the list of initialized GPUs (i.e mInitDoneGpuInfoMap)
    FMAutoLock lock(mLock);

    FM_LOG_DEBUG("entering initializeAllGpus");

    //
    // bail out if GPUs are already attached. This is supposed to be
    // called during initialization time
    //
    if (0 != mInitDoneGpuInfoMap.size()) {
        FM_LOG_WARNING("request to initialize all GPUs when some GPUs are already initialized");
        return FM_INT_ST_GENERIC_ERROR;
    }
    // get list of gpus and its IDs from RM
    getProbedGpuCount(gpuIdList);
    FM_LOG_DEBUG("probed GPU count: %d\n", (int)gpuIdList.size());

    // attach each probed GPUs
    for (it = gpuIdList.begin(); it != gpuIdList.end(); it++) {
        LwU32 gpuId = (*it);
        FM_LOG_DEBUG("Initializing GPU probe ID = %d", gpuId);
        // get the pci bdf information
        FMPciInfo_t pciInfo;
        fmResult = getGpuPciInfo(gpuId, pciInfo);
        if (FM_INT_ST_OK != fmResult) {
            // error already logged. To support degraded mode and ignore failed GPUs,
            // continue with rest of the GPUs. This can also happen if a GPU is
            // unbound from the driver in the meantime.
            FM_LOG_WARNING("unable to get pci bus id information, skipping probed GPU ID: %d", gpuId);
            continue;
        }
        // get gpu uuid information
        FMUuid_t uuid;
        fmResult = getGpuUuidInfo(gpuId, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // UUID query may not succeed for all GPUs (like consumer GPUs/old VBIOS), skip them
            FM_LOG_WARNING("unable to get UUID, skipping GPU pci bus id: %s probed ID: %d",
                           pciInfo.busId, gpuId);
            continue;
        }

        // attach and open all required GPU handles
        fmResult = attachAndOpenGpuHandles(gpuId, pciInfo, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // error already logged and clean-up done. To support degraded mode and
            // ignore failed GPUs, continue with rest of the GPUs.
            continue;
        }
    }
    FM_LOG_DEBUG("attached GPU count: %d\n", (int)mInitDoneGpuInfoMap.size());

    return FM_INT_ST_OK;
}

/******************************************************************************************
 Method to detach(deinitialize) all GPUs. Used by LocalFM during shutdown time or after
 shared LWSwitch model initialization
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::deInitializeAllGpus()
{
    FM_LOG_DEBUG("entering deInitializeAllGpus");
    // hold lock as this will modify the list of initialized GPUs (i.e mInitDoneGpuInfoMap)
    FMAutoLock lock(mLock);

    if (0 == mInitDoneGpuInfoMap.size()) {
        FM_LOG_WARNING("request to deinitialize all GPUs when no GPUs are initialized");
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
    mFdToGpuUuidMap.clear();
    mEventDescriptorList.clear();
    mEventSubscriberInfoMap.clear();
    return FM_INT_ST_OK;
}

/******************************************************************************************
 Method to attach(initialize) a single GPU. Used by LocalFM during partition activation
 in shared LWSwitch model
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::initializeGpu(FMUuid_t &uuidToInit, bool registerEvt)
{
    FMIntReturn_t fmResult;
    // hold lock as this will modify the list of initialized GPUs (i.e mInitDoneGpuInfoMap)
    FMAutoLock lock(mLock);

    FM_LOG_DEBUG("entering initializeGpu");
    // check whether the GPU is already attached and initialized.
    InitDoneGpuInfoMap_t::iterator initDoneIt = mInitDoneGpuInfoMap.find(uuidToInit);
    if (initDoneIt != mInitDoneGpuInfoMap.end()) {
        FM_LOG_ERROR("trying to initialize an already initialized GPU, uuid: %s", uuidToInit.bytes);
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
            // error already logged. To support degraded mode and ignore failed GPUs,
            // continue with rest of the GPUs. This can also happen if a GPU is
            // unbound from the driver in the meantime.
            FM_LOG_WARNING("unable to get pci bus id information, skipping probed GPU ID: %d", gpuId);
            continue;
        }
        // get gpu uuid information
        FMUuid_t uuid;
        fmResult = getGpuUuidInfo(gpuId, uuid);
        if (FM_INT_ST_OK != fmResult) {
            // UUID query may not succeed for all GPUs (like consumer GPUs/old VBIOS), skip them
            FM_LOG_WARNING("unable to get UUID, skipping GPU pci bus id: %s probed ID: %d",
                           pciInfo.busId, gpuId);
            continue;
        }

        // attach if the uuid matches
        if (strncmp(uuidToInit.bytes, uuid.bytes, FM_UUID_BUFFER_SIZE) == 0) {
            // attach and open all required GPU handles
            // error already logged and clean-up done.
            fmResult = attachAndOpenGpuHandles(gpuId, pciInfo, uuid);
            if (FM_INT_ST_OK != fmResult) {
                return fmResult;
            }

            if (registerEvt == true) {
                if (FM_INT_ST_OK != registerEventForGpu(uuidToInit)) {
                    FM_LOG_WARNING("failed to register event for GPU, uuid: %s",
                                   uuidToInit.bytes);
                }
            }
            return fmResult;
        }
    }

    // not found the required GPU/uuid
    FM_LOG_ERROR("request to attach/open GPU uuid: %s failed as the GPU is not found in the probed list",
                 uuidToInit.bytes);
    return FM_INT_ST_BADPARAM;
}

/******************************************************************************************
 Method to detach(deinitialize) a single GPU. Used by LocalFM during partition deactivation
 in shared LWSwitch model
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::deInitializeGpu(FMUuid_t &uuidToDeinit, bool unRegisterEvt)
{
    FMIntReturn_t fmResult;

    // hold lock as this will modify the list of initialized GPUs (i.e mInitDoneGpuInfoMap)
    FMAutoLock lock(mLock);

    FM_LOG_DEBUG("entering deInitializeGpu");
    // check whether the GPU is already attached and initialized.
    InitDoneGpuInfoMap_t::iterator initDoneIt = mInitDoneGpuInfoMap.find(uuidToDeinit);
    if (initDoneIt == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to deinitialize a GPU which is not initialized, uuid: %s", uuidToDeinit.bytes);
        return FM_INT_ST_BADPARAM;
    }
    FMUuid_t uuid = initDoneIt->first;
    GpuCtxInfo_t &gpuCtxInfo = initDoneIt->second;

    if (unRegisterEvt == true) {
        fmResult = unRegisterEventForGpu(uuidToDeinit);
        if (FM_INT_ST_OK != fmResult) {
            FM_LOG_WARNING("failed to unregister event for GPU, uuid: %s",
                           uuidToDeinit.bytes);
        }
    }

    // close all the device and sub device handles
    detachAndCloseGpuHandles(gpuCtxInfo, uuid);
    // remove the specified GPU from init done map
    mInitDoneGpuInfoMap.erase(initDoneIt);
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getInitializedGpuInfo(FMGpuInfoList &initDoneGpuInfo)
{
    // hold lock as this will access the list of initialized GPUs (i.e mInitDoneGpuInfoMap)
    FMAutoLock lock(mLock);

    initDoneGpuInfo.clear();
    InitDoneGpuInfoMap_t::iterator it;
    for (it = mInitDoneGpuInfoMap.begin(); it != mInitDoneGpuInfoMap.end(); it++ ) {
        FMUuid_t uuid = it->first;
        GpuCtxInfo_t &gpuCtxInfo = it->second;
        FMGpuInfo_t gpuInfo;
        gpuInfo.gpuIndex = gpuCtxInfo.deviceInstance;
        gpuInfo.uuid = uuid;
        gpuInfo.pciInfo = gpuCtxInfo.pciInfo;
        gpuInfo.discoveredLinkMask = gpuCtxInfo.discoveredLinkMask;
        gpuInfo.enabledLinkMask = gpuCtxInfo.enabledLinkMask;
        gpuInfo.archType = gpuCtxInfo.archType;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        gpuInfo.isEgmCapable = gpuCtxInfo.isEgmCapable;
        gpuInfo.isSpaCapable = gpuCtxInfo.isSpaCapable;
        gpuInfo.spaAddress = gpuCtxInfo.spaAddress;
#endif
        initDoneGpuInfo.push_back(gpuInfo);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getExcludedGpuInfo(std::list<FMExcludedGpuInfo_t> &excludedGpuInfo)
{
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probeParams = {{0}};
    LwU32 rmResult;
    excludedGpuInfo.clear();
    // get GPUs detected/probed by RM
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle,
                           LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probeParams, sizeof(probeParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to get list of probed GPUs from LWPU GPU driver failed with error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    for (LwU32 idx = 0; idx < LW0000_CTRL_GPU_MAX_PROBED_GPUS; idx++) {
        if (probeParams.excludedGpuIds[idx] != LW0000_CTRL_GPU_ILWALID_ID) {
            LwU32 gpuId = probeParams.excludedGpuIds[idx];
            FMExcludedGpuInfo_t gpuInfo;
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
                // UUID query for excluded GPUs should succeed as RM uses UUID to exclude GPUs.
                // error already logged
                return fmResult;
            }
            // got all the information for this GPU
            excludedGpuInfo.push_back(gpuInfo);
        }
    }
    return FM_INT_ST_OK;
}

LwU32
LocalFMGpuMgr::getProbedGpuCount()
{
    std::vector<LwU32> gpuIdList;
    getProbedGpuCount(gpuIdList);
    return gpuIdList.size();
}

FMIntReturn_t
LocalFMGpuMgr::allocFmSession(unsigned int abortLwdaJobsOnFmExit)
{
    LwU32 rmResult;
    if (0 != mFMSessionHandle) {
        FM_LOG_ERROR("fabric manager session for LWPU GPU driver is already allocated");
        return FM_INT_ST_GENERIC_ERROR;
    }

    LwHandle fmSession;
    if (!FMHandleGenerator::allocHandle(fmSession)) {
        FM_LOG_DEBUG("allocating fabric manager session handle failed");
        return FM_INT_ST_GENERIC_ERROR;
    }

    LW000F_ALLOCATION_PARAMETERS fmSessionAllocParams = {0};
    if (abortLwdaJobsOnFmExit == 1) {
        //
        // need to abort lwca jobs on FM exit. this means allow RM to enable RC channel
        // recovery/close on FMSession clean-up
        //
        fmSessionAllocParams.flags = LW000F_FLAGS_CHANNEL_RECOVERY_ENABLED;
    } else {
        //
        // let the lwca jobs continue on FM exit. this means disable RC channel
        // recovery/close on FMSession clean-up
        //
        fmSessionAllocParams.flags = LW000F_FLAGS_CHANNEL_RECOVERY_DISABLED;
    }

    // Alloc FM session, pass in FM session handle.
    rmResult = LwRmAlloc(mRmClientHandle, mRmClientHandle, fmSession,
                         FABRIC_MANAGER_SESSION, &fmSessionAllocParams);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to allocate fabric manager session for LWPU GPU driver, error: %s",
                     lwstatusToString(rmResult));
        FMHandleGenerator::freeHandle(fmSession);
        return FM_INT_ST_GENERIC_ERROR;
    }
    mFMSessionHandle = fmSession;
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::freeFmSession()
{
    if (0 == mFMSessionHandle) {
        FM_LOG_ERROR("fabric manager session for LWPU GPU driver is not allocated");
        return FM_INT_ST_GENERIC_ERROR;
    }
    LwRmFree(mRmClientHandle, mRmClientHandle, mFMSessionHandle);
    mFMSessionHandle = 0;
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::fmSessionSetState()
{
    LwU32 rmResult;
    if (0 == mFMSessionHandle) {
        FM_LOG_ERROR("fabric manager session for LWPU GPU driver is not allocated");
        return FM_INT_ST_GENERIC_ERROR;
    }
    rmResult = LwRmControl(mRmClientHandle, mFMSessionHandle, LW000F_CTRL_CMD_SET_FM_STATE, NULL, 0);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to set fabric manager session in LWPU GPU driver, error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::fmSessionSetNodeId(uint32 nodeId)
{
    LwU32 rmResult;
    if (0 == mFMSessionHandle) {
        FM_LOG_ERROR("fabric manager session for LWPU GPU driver is not allocated");
        return FM_INT_ST_GENERIC_ERROR;
    }

    LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS fmSessionNodeIdParams;
    fmSessionNodeIdParams.nodeId = nodeId;

    rmResult = LwRmControl(mRmClientHandle, mFMSessionHandle, LW000F_CTRL_CMD_SET_FABRIC_NODE_ID,
                           &fmSessionNodeIdParams, sizeof(fmSessionNodeIdParams));
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to set fabric manager session " NODE_ID_LOG_STR " in LWPU GPU driver, error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::fmSessionClearState()
{
    LwU32 rmResult;
    if (0 == mFMSessionHandle) {
        FM_LOG_ERROR("fabric manager session for LWPU GPU driver is not allocated");
        return FM_INT_ST_GENERIC_ERROR;
    }
    rmResult = LwRmControl(mRmClientHandle, mFMSessionHandle, LW000F_CTRL_CMD_CLEAR_FM_STATE, NULL, 0);
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to clear fabric manager session in LWPU GPU driver, error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::setGpuFabricGPA(FMUuid_t gpuUuid, unsigned long long baseAddress)
{
    LW2080_CTRL_GPU_SET_FABRIC_BASE_ADDR_PARAMS addressParams = {0};
    LwU32 rmResult;
    FMIntReturn_t fmResult;

    // check whether the GPU is initialized before setting base address
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to set base address for an uninitialized GPU, uuid: %s count of GPUs %d",
                        gpuUuid.bytes, (int)mInitDoneGpuInfoMap.size() );
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;
    if (gpuCtxInfo.enabledLinkMask == 0) {
        // There is no LWLink enabled, no need to configure GPA
        // This could happen on MIG enabled GPU
        return FM_INT_ST_OK;
    }
    addressParams.fabricBaseAddr = baseAddress;
    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR, &addressParams, sizeof(addressParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to set base address for GPU uuid: %s pci bus id: %s failed with error: %s",
                     gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
FMIntReturn_t
LocalFMGpuMgr::setGpuFabricGPAEgm(FMUuid_t gpuUuid, unsigned long long egmBaseAddress)
{
    LW2080_CTRL_GPU_SET_EGM_GPA_FABRIC_BASE_ADDR_PARAMS addressParams = {0};
    LwU32 rmResult;
    FMIntReturn_t fmResult;

    // check whether the GPU is initialized before setting base address
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to set EGM base address for an uninitialized GPU, uuid: %s count of GPUs %d",
                        gpuUuid.bytes, (int)mInitDoneGpuInfoMap.size() );
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;
    if (gpuCtxInfo.enabledLinkMask == 0) {
        // There is no LWLink enabled, no need to configure GPA EGM address
        // This could happen on MIG enabled GPU
        return FM_INT_ST_OK;
    }
    addressParams.egmGpaFabricBaseAddr = egmBaseAddress;
    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_GPU_SET_FABRIC_BASE_ADDR, &addressParams, sizeof(addressParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to set EGM base address for GPU uuid: %s pci bus id: %s failed with error: %s",
                     gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}
#endif

FMIntReturn_t
LocalFMGpuMgr::setGpuFabricFLA(FMUuid_t gpuUuid, unsigned long long baseAddress,
                               unsigned long long size)
{
    LW2080_CTRL_FLA_RANGE_PARAMS addressParams = {0};
    LwU32 rmResult;
    FMIntReturn_t fmResult;

    // check whether the GPU is initialized before setting FLA address
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to set fabric linear address for an uninitialized GPU, uuid: %s", gpuUuid.bytes);
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;
    if (gpuCtxInfo.enabledLinkMask == 0) {
        // There is no LWLink enabled, no need to configure FLA
        // This could happen on MIG enabled GPU
        return FM_INT_ST_OK;
    }

    addressParams.base = baseAddress;
    addressParams.size = size;
    addressParams.mode = LW2080_CTRL_FLA_RANGE_PARAMS_MODE_INITIALIZE;
    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_FLA_RANGE, &addressParams, sizeof(addressParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to set fabric linear address for GPU uuid: %s pci bus id: %s failed with error: %s",
                     gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::clearGpuFabricFLA(FMUuid_t gpuUuid, unsigned long long baseAddress,
                                 unsigned long long size)
{
    LW2080_CTRL_FLA_RANGE_PARAMS addressParams = {0};
    LwU32 rmResult;
    FMIntReturn_t fmResult;

    // check whether the GPU is initialized before setting FLA address
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to clear fabric linear address for an uninitialized GPU, uuid: %s", gpuUuid.bytes);
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;

    addressParams.base = baseAddress;
    addressParams.size = size;
    addressParams.mode = LW2080_CTRL_FLA_RANGE_PARAMS_MODE_DESTROY;
    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_FLA_RANGE, &addressParams, sizeof(addressParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to clear fabric linear address for GPU uuid: %s pci bus id: %s failed with error: %s",
                     gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
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
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle,
                           LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probeParams, sizeof(probeParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to get list of probed GPUs from LWPU GPU driver failed with error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    for (LwU32 idx = 0; idx < LW0000_CTRL_GPU_MAX_PROBED_GPUS; idx++) {
        if (probeParams.gpuIds[idx] != LW0000_CTRL_GPU_ILWALID_ID) {
            gpuIds.push_back(probeParams.gpuIds[idx]);
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
    FM_LOG_DEBUG("attaching and opening GPU pci bus id: %s ID: %d uuid: %s",
                  gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, uuid.bytes);
    // skip drain state enabled GPUs
    if (isGpuInDrainState(gpuCtxInfo.gpuId)) {
        FM_LOG_WARNING("not attaching/opening drain state enabled GPU pci bus id: %s probed ID: %d uuid: %s",
                       gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, uuid.bytes);
        return FM_INT_ST_OK;
    }
    // attach the GPU
    fmResult = doRmAttachGpu(gpuCtxInfo.gpuId);
    if (FM_INT_ST_OK != fmResult) {
        FM_LOG_ERROR("unable to open/attach GPU handle for GPU pci bus id: %s probed ID: %d uuid: %s",
                     gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, uuid.bytes);
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

    //
    // check whether the GPU architecture is supported by FM. The GPU architecture
    // information can't be queried without attaching/opening it. So, detach/close it
    // if the architecture type is not supported by FM.
    //
    if (false == isGpuArchitectureSupported(gpuCtxInfo)) {
        // error already logged, do the clean-up
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
        return FM_INT_ST_NOT_SUPPORTED;
    }

    //
    // get the supported and enabled link mask information for the GPU
    // for MIG enabled GPU, enabled link mask will be zero.
    //
    fmResult = getGpuLWLinkCapInfo(gpuCtxInfo);
    if (FM_INT_ST_OK != fmResult) {
        // error already logged, do the clean-up
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
        return fmResult;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // get the GPU EGM capability
    fmResult = getGpuCapabilityInfo(gpuCtxInfo);
    if (FM_INT_ST_OK != fmResult) {
        // error already logged, do the clean-up
        detachAndCloseGpuHandles(gpuCtxInfo, uuid);
        return fmResult;
    }
#endif

    // all looks good. keep/cache the information in our context
    mInitDoneGpuInfoMap.insert(std::make_pair(uuid, gpuCtxInfo));

    FM_LOG_DEBUG("GPU pci bus id: %s ID: %d uuid: %s is attached.",
                  gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, uuid.bytes);
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::detachAndCloseGpuHandles(GpuCtxInfo_t &gpuCtxInfo, FMUuid_t &uuid)
{
    FM_LOG_DEBUG("detachAndCloseGpuHandles for GPU pci bus id: %s", gpuCtxInfo.pciInfo.busId);
    freeGpuSubDevices(gpuCtxInfo);
    doRmDeatchGpu(gpuCtxInfo.gpuId);
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuPciInfo(LwU32 gpuId, FMPciInfo_t &pciInfo)
{
    LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS pciInfoParams = {0};
    LwU32 rmResult;
    FM_LOG_DEBUG("Getting PCI info for GPU ID = %d", gpuId);
    pciInfoParams.gpuId = gpuId;
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle,
                           LW0000_CTRL_CMD_GPU_GET_PCI_INFO, &pciInfoParams, sizeof(pciInfoParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to get PCI information from LWPU GPU driver for probed GPU ID: %d, error: %s",
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
    FM_LOG_DEBUG("Getting UUID info for GPU ID = %d", gpuId);
    uuidParams.gpuId = gpuId;
    uuidParams.flags = LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_ASCII;
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID,
                           &uuidParams, sizeof(uuidParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to get UUID information from LWPU GPU driver for probed GPU ID: %d, error:%s",
                     gpuId,  lwstatusToString(rmResult));
#ifdef DEBUG
        // On early systems and bringup boards, there might be no UUID set
        // for the GPU, add Fake ones here, faking gpu UUID in the debug build
        snprintf(uuidInfo.bytes, sizeof(uuidInfo.bytes), "%d", gpuId);
        FM_LOG_ERROR( "Fake GPU ID = %d UUID = %s\n", gpuId, uuidInfo.bytes);
        return FM_INT_ST_OK;
#else
        return FM_INT_ST_GENERIC_ERROR;
#endif
    }
    memset(uuidInfo.bytes, 0, sizeof(uuidInfo.bytes));
    strncpy(uuidInfo.bytes, (char *)uuidParams.gpuUuid, FM_UUID_BUFFER_SIZE - 1);
    FM_LOG_DEBUG("GPU ID = %d UUID = %s", gpuId, uuidInfo.bytes);
    return FM_INT_ST_OK;
}

bool
LocalFMGpuMgr::isGpuInDrainState(LwU32 gpuId)
{
    LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS drainParms = {0};
    LwU32 rmResult;
    bool drainState = false;
    drainParms.gpuId = gpuId;
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE,
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
FMIntReturn_t
LocalFMGpuMgr::getGpuCapabilityInfo(GpuCtxInfo_t &gpuCtxInfo)
{
    LW2080_CTRL_GPU_GET_INFO_V2_PARAMS params = {0};

    LwU32 rmResult;
    gpuCtxInfo.isEgmCapable = false;
    gpuCtxInfo.isSpaCapable = false;

    params.gpuInfoListSize = 2;
    params.gpuInfoList[0].index = LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY;
    params.gpuInfoList[1].index = LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY;


    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle, LW2080_CTRL_CMD_GPU_GET_INFO_V2,
                           &params, sizeof(params));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to get EGM information from LWPU GPU driver for probed GPU ID: %d, error:%s",
                     gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    if (params.gpuInfoList[0].data  == LW2080_CTRL_GPU_INFO_INDEX_GPU_LOCAL_EGM_CAPABILITY_YES) {
        gpuCtxInfo.isEgmCapable = true;
    }

    if (params.gpuInfoList[1].data  == LW2080_CTRL_GPU_INFO_INDEX_GPU_ATS_CAPABILITY_NO) {
        return FM_INT_ST_OK;
    }

    // TODO when the RM API is ready, get the GPU SPA address

    gpuCtxInfo.isSpaCapable = true;
    return FM_INT_ST_OK;
}
#endif

FMIntReturn_t
LocalFMGpuMgr::doRmAttachGpu(LwU32 gpuId)
{
    LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams = {{0}};
    LwU32 rmResult;
    attachParams.gpuIds[0] = gpuId;
    attachParams.gpuIds[1] = LW0000_CTRL_GPU_ILWALID_ID;
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_ATTACH_IDS,
                           &attachParams, sizeof(attachParams));
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to open/attach handle to probed GPU ID: %d, error: %s",
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
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_DETACH_IDS,
                           &detachParams, sizeof(detachParams));
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to close/detach handle to probed GPU ID: %d, error: %s",
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
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_GET_ID_INFO,
                           &idInfoParams, sizeof(idInfoParams));
    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to get device instance id information for GPU id: %d, error: %s",
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
    LwU32 lw0080handle;

    if (!FMHandleGenerator::allocHandle(lw0080handle)) {
        FM_LOG_DEBUG("allocating a handle failed");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmAlloc(mRmClientHandle, mRmClientHandle, lw0080handle,
                         LW01_DEVICE_0, &allocDeviceParams);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to allocate GPU base device instance for probed GPU ID: %d error: %s",
                     gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    // cache the handle after succesful allocation
    gpuCtxInfo.lw0080handle = lw0080handle;
    // allocate sub devices
    LW2080_ALLOC_PARAMETERS allocSubDeviceParams = {0};
    allocSubDeviceParams.subDeviceId = 0;
    LwU32 lw2080handle;

    if (!FMHandleGenerator::allocHandle(lw2080handle)) {
        FM_LOG_DEBUG("allocating a handle failed");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmAlloc(mRmClientHandle, gpuCtxInfo.lw0080handle, lw2080handle,
                         LW20_SUBDEVICE_0, &allocSubDeviceParams);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to allocate GPU sub device instance for probed GPU ID: %d error: %s",
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
    LwU32 rmResult;

    if (gpuCtxInfo.lw2080handle) {
        rmResult = LwRmFree(mRmClientHandle, gpuCtxInfo.lw2080handle, gpuCtxInfo.lw2080handle);
        if (LWOS_STATUS_SUCCESS != rmResult) {
            FM_LOG_ERROR("request to free LWLink subsystem handle failed for pci bus id: %s probed ID: %d with error: %s",
                         gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        }
        FMHandleGenerator::freeHandle(gpuCtxInfo.lw2080handle);
        gpuCtxInfo.lw2080handle = 0;
    }
    if (gpuCtxInfo.lw0080handle) {
        rmResult = LwRmFree(mRmClientHandle, gpuCtxInfo.lw0080handle, gpuCtxInfo.lw0080handle);
        if (LWOS_STATUS_SUCCESS != rmResult) {
            FM_LOG_ERROR("request to free control subsystem handle failed for pci bus id: %s probed ID: %d with error: %s",
                         gpuCtxInfo.pciInfo.busId, gpuCtxInfo.gpuId, lwstatusToString(rmResult));
        }
        FMHandleGenerator::freeHandle(gpuCtxInfo.lw0080handle);
        gpuCtxInfo.lw0080handle = 0;
    }
    return FM_INT_ST_OK;
}

LwHandle
LocalFMGpuMgr::getMemSubDeviceHandleForGpuId(LwU32 gpuId)
{
    //Walk the mInitDoneGpuInfoMap and print all values in it-second.
    for (InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.begin(); it != mInitDoneGpuInfoMap.end(); ++it) {
        GpuCtxInfo_t gpuCtxInfo = it->second;
        if (gpuId == gpuCtxInfo.gpuId)
            return gpuCtxInfo.lw2080handle;
    }
    return 0;
}

LwHandle
LocalFMGpuMgr::getRmClientHandle()
{
    return mRmClientHandle;
}

LwHandle
LocalFMGpuMgr::getFmSessionHandle()
{
    return mFMSessionHandle;
}
FMIntReturn_t
LocalFMGpuMgr::setGpuLWLinkInitDisabledMask(FMUuid_t uuid, unsigned int disabledMask)
{
    FMIntReturn_t fmResult;
    // query RM to get a list of lwrrently probed GPUs.
    std::vector<LwU32> gpuIdList;
    std::vector<LwU32>::iterator it;

    getProbedGpuCount(gpuIdList);
    if (gpuIdList.empty()) {
        FM_LOG_ERROR("list of probed GPUs by LWPU GPU Driver is empty while trying to set enabled LWLink mask");
        return FM_INT_ST_GENERIC_ERROR;
    }

    for (it = gpuIdList.begin(); it != gpuIdList.end(); it++) {
        LwU32 gpuId = (*it);

        // get gpu uuid information
        FMUuid_t gpuUuid;
        fmResult = getGpuUuidInfo(gpuId, gpuUuid);
        if (FM_INT_ST_OK != fmResult) {
            // UUID query may not succeed for all GPUs (like consumer GPUs/old VBIOS), skip them
            FM_LOG_WARNING("failed to get UUID, skipping probed GPU ID: %d", gpuId);
            continue;
        }

        if (uuid == gpuUuid) {
            return setInitDisabledLinkMask(gpuId, disabledMask);
        }
    }

    // this means we didn't find the specified GPU uuid in the lwrrently probed list
    FM_LOG_ERROR("request to set enabled LWLink mask failed for GPU UUID: %s as it is not found in the probed list",
                  uuid.bytes);
    return FM_INT_ST_BADPARAM;
}

/******************************************************************************************
 * This is used in conjunction with LWSwitch Shared Virtualization feature where GPUs are
 * hot-plugged to Service VM/RM (by Hypervisor) and Fabric Manager is signaled
 * externally by the Hypervisor to initialize those GPUs. Without this, GPUs attached to
 * RM after RMLib client initializations will not be accessible and all LWRM calls will
 * fail on them.
 *
 * This is a temporary work around to forcefully notify the RMLib layer to re-fetch the
 * probed GPU information and update its local context information
 *
 ******************************************************************************************/
FMIntReturn_t
LocalFMGpuMgr::refreshRmLibProbedGpuInfo()
{
#ifdef __linux__
    LwU32 rmResult;
    FMIntReturn_t fmResult;

    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_OS_UNIX_REFRESH_RMAPI_DEVICE_LIST, NULL, 0);

    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_WARNING("request to refresh GPU API layer with probed GPU information failed with error:%s",
                        lwstatusToString(rmResult));
         return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
#endif

    //
    // TODO: for Windows, this may not be required as plug and play is supported
    // so returning OK for now
    //
    return FM_INT_ST_OK;
}

void
LocalFMGpuMgr::run()
{
    FMIntReturn_t fmResult;

    /* both subscribers have subscribed to the service, now we start polling */
    while (!ShouldStop()) {
        /* poll for errors for fm stats reporter */
        /* watch events here and in watch events, call the corresponding callback */
        fmResult = watchEvents(FATAL_ERROR_POLLING_TIMEOUT * 1000);
        if (fmResult == FM_INT_ST_BREAK) {
            break;
        }
        lwosThreadYield();
    }
    return;
}

FMIntReturn_t
LocalFMGpuMgr::subscribeForGpuEvents(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    //
    // This lock is required as there is a chance that two subscribers
    // can subscribe at the same time and edit the mEventSubscriberinfo map
    //
    FMAutoLock lock(mLock);

    /* register events and callback */
    addSubscriberInfoToEventList(eventVal, callback, subscriberCtx);
    fmResult = registerEvent(eventVal);
    return fmResult;
}

FMIntReturn_t
LocalFMGpuMgr::unsubscribeGpuEvents(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    //
    // This lock is required as there is a chance that two subscribers
    // can unsubscribe or subscribe at the same time and edit the mEventSubscriberinfo map
    //
    FMAutoLock lock(mLock);

    fmResult = unRegisterEvent(eventVal, callback, subscriberCtx);

    /* This check is to see if all the events from the all the subscribers have been unsubscribed */
    if (mEventSubscriberInfoMap.size()) {
        return fmResult;
    }

    /*
        Code comes here only when all the events from all the subscribers
        have been unsubscribed. All file descriptor data structures are cleared below.
    */

    /*clear event subscriber info map */
    mEventDescriptorList.clear();
    mFdToGpuUuidMap.clear();

    return fmResult;
}

FMIntReturn_t
LocalFMGpuMgr::startGpuEventWatch()
{
    // create and kick start the FMThread run() loop for event watching
    if (0 != Start()) {
        // failed to create or start the thread
        return FM_INT_ST_GENERIC_ERROR;
    }
    return FM_INT_ST_OK;
}

void
LocalFMGpuMgr::stopGpuEventWatch()
{
    int st = StopAndWait((FATAL_ERROR_POLLING_TIMEOUT + 1) * 1000);
    if (st) {
        FM_LOG_WARNING("fm gpu manager: killing thread after stop request timeout");
        Kill();
    }
}

FMIntReturn_t
LocalFMGpuMgr::registerEvent(LwU32 eventVal)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    switch (eventVal) {
        /*
            For GPU Error Events:
            i)  For each GPU, register an OS Event
            ii) For each type of GPU Error per GPU, allocate an LWEvent and set notification for event
        */
        case LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED:
        case LW2080_NOTIFIERS_LWLINK_ERROR_FATAL:
            {
                if (mFdToGpuUuidMap.size() == 0) {
                    fmResult = allocOsEventForGpus();
                    if (fmResult != FM_INT_ST_OK)
                        return fmResult;
                }
                fileDescriptorToGpuUuidMap_t::iterator it;
                for (it = mFdToGpuUuidMap.begin(); it != mFdToGpuUuidMap.end(); it++) {
                    int fd = it->first;
                    FMUuid_t gpuUuid = it->second;
                    LwU32 lw2080handle = mInitDoneGpuInfoMap[gpuUuid].lw2080handle;
                    fmResult = allocEventForGpuErrorType(eventVal, fd, lw2080handle);
                    if (fmResult != FM_INT_ST_OK) {
                        return fmResult;
                    }
                }
            }
            break;
        case LW000F_NOTIFIERS_FABRIC_EVENT:
            {
                //
                // Fabric event is actually registered multiple times, once by LocalMemMgr and
                // by LocalMemMgrImporter. But it is enough if we register the event once with
                // RM. There is no need to create two seperate OS events. If the event for
                // fabric events fd is fired, then all subscribers for the events fd (ie LocalMemMgr
                // and LocalMemMgrImporter) will be notified.
                //
                if (mFabricEventsFd == -1) {
                    allocRmEvent();
                }
            }
            break;
    }

    return fmResult;
}

//This function need not hold lock as the caller function is already holding it
FMIntReturn_t
LocalFMGpuMgr::allocRmEvent()
{
    LwHandle handleOsEvent;
    LwU32 retVal;
    int fd;

    retVal = LwRmAllocOsEvent(mRmClientHandle, 0, &handleOsEvent, &fd);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        FM_LOG_ERROR("GPU Driver event allocation failed with error 0x%x", retVal);
        return FM_INT_ST_GENERIC_ERROR;
    }

    LW0005_ALLOC_PARAMETERS params;
    memset(&params, 0, sizeof(params));
    params.hParentClient = mRmClientHandle;
    params.hSrcResource = mFMSessionHandle;
    params.hClass = LW01_EVENT_OS_EVENT;
    // the reason we are adding the LW01_EVENT_WITHOUT_EVENT_DATA flag is to
    // optimize FM event processing by preventing a control call to RM to
    // get event data type (for fabric events)
    params.notifyIndex = LW000F_NOTIFIERS_FABRIC_EVENT | LW01_EVENT_WITHOUT_EVENT_DATA;
    params.data = (LwP64)(&fd);

    retVal = LwRmAlloc(mRmClientHandle, mFMSessionHandle, mFabricEventsHandle, LW01_EVENT, &params);
    if (retVal != LWOS_STATUS_SUCCESS)
    {
        FM_LOG_ERROR("request to allocate fabric manager GPU Driver event handler failed with error 0x%x", retVal);
        return FM_INT_ST_GENERIC_ERROR;
    }

    // caching the fabric events fd in a member variable so that after polling, if this
    // event is fired, we can selectively skip calling the RM control call for getting
    // event data
    mFabricEventsFd = fd;
    mEventDescriptorList.push_back(fd);

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::allocEventForGpuErrorType(LwU32 eventVal, int &fd, LwU32 lw2080handle)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    LwU32 rmResult;

    int oldfd = fd;

    LwU32 handle;

    if (!FMHandleGenerator::allocHandle(handle)) {
        FM_LOG_DEBUG("allocating sub device handle for GPU failed");
        return FM_INT_ST_GENERIC_ERROR;
    }

    rmResult = LwRmAllocEvent(mRmClientHandle, lw2080handle, handle, LW01_EVENT_OS_EVENT,
                             eventVal, (void *) &fd);

    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to allocate GPU Driver event object for GPU error monitoring, error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    if(oldfd != fd) {
        LwRmFree(mRmClientHandle, lw2080handle, handle);
        return FM_INT_ST_GENERIC_ERROR;
    }

    LW2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS params = {0};
    params.event  = eventVal;
    params.action = LW2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT;
    rmResult = LwRmControl(mRmClientHandle, lw2080handle, LW2080_CTRL_CMD_EVENT_SET_NOTIFICATION, &params, sizeof(params));
    if (LWOS_STATUS_SUCCESS != rmResult)
    {
        LwRmFree(mRmClientHandle, lw2080handle, handle);
        FM_LOG_ERROR("request to set GPU Driver event notification action type failed with error: %s",
                     lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

//This function need not hold lock as the caller function is already holding it
FMIntReturn_t
LocalFMGpuMgr::allocOsEventForGpus()
{
    LwU32 device;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    LwU32 rmResult;
    LwHandle hOsEvent;
    int fd, gpuId;
    InitDoneGpuInfoMap_t::iterator it;

    for (it = mInitDoneGpuInfoMap.begin(); it != mInitDoneGpuInfoMap.end(); it++ ) {
        FMUuid_t gpuUuid = it->first;
        GpuCtxInfo_t &gpuCtxInfo = it->second;
        LwU32 lw2080handle = gpuCtxInfo.lw2080handle;
        LwU32 gpuIndex = gpuCtxInfo.gpuId;
        rmResult = LwRmAllocOsEvent(mRmClientHandle, lw2080handle, NULL, &fd);
        if(rmResult != LWOS_STATUS_SUCCESS) {
            FM_LOG_ERROR("request to allocate GPU Driver event monitoring object for probed GPU ID: %d failed with error: %s",
                         (int) gpuIndex, lwstatusToString(rmResult));
            return FM_INT_ST_GENERIC_ERROR;
        }
        mFdToGpuUuidMap[fd] = gpuUuid;
        mEventDescriptorList.push_back(fd);
    }
    return fmResult;
}

FMIntReturn_t
LocalFMGpuMgr::freeOsEvent(int fd, LwU32 lw2080handle)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    LwU32 rmResult;

    rmResult = LwRmFreeOsEvent(mRmClientHandle, lw2080handle, (LwU32) fd);

    if(rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("request to free GPU Driver event allocated for error monitoring failed with error: %s",
                     lwstatusToString(rmResult));
        fmResult = FM_INT_ST_GENERIC_ERROR;
    }

    return fmResult;
}

FMIntReturn_t
LocalFMGpuMgr::unRegisterGpuErrorEvents()
{
    fileDescriptorToGpuUuidMap_t::iterator it;
    eventDescriptorList_t::iterator fdIt = mEventDescriptorList.begin();
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    for (it = mFdToGpuUuidMap.begin(); it != mFdToGpuUuidMap.end(); it++ ) {
        int fd = it->first;
        FMUuid_t gpuUuid = it->second;
        if (mInitDoneGpuInfoMap.find(gpuUuid) == mInitDoneGpuInfoMap.end()) {
            return FM_INT_ST_BADPARAM;
        }
        GpuCtxInfo_t gpuCtxInfo = mInitDoneGpuInfoMap[gpuUuid];
        LwU32 lw2080handle = gpuCtxInfo.lw2080handle;
        LwU32 gpuIndex = gpuCtxInfo.gpuId;

        fmResult = freeOsEvent(fd, lw2080handle);

        if (fmResult != FM_INT_ST_OK) {
            // error already logged
            return fmResult;
        }

        /* Clear the fds in the event descriptor list as these need not be watched anymore */
        if (*fdIt == fd) {
            fdIt = mEventDescriptorList.erase(fdIt);
            continue;
        }
        fdIt++;
    }
    mFdToGpuUuidMap.clear();

    return fmResult;
}

FMIntReturn_t
LocalFMGpuMgr::unRegisterEvent(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    eventSubscriberInfoMap_t::iterator it;
    if ((it = mEventSubscriberInfoMap.find(eventVal)) == mEventSubscriberInfoMap.end()) {
        FM_LOG_ERROR("trying to unregister an error monitoring event type that has not been registered");
        return FM_INT_ST_BADPARAM;
    }

    subscriberInfoList_t::iterator subscriberIt;

    for (subscriberIt = it->second.begin(); subscriberIt != it->second.end(); subscriberIt++) {
        subscriberInfo_t subscriber = *subscriberIt;
        if (subscriber.subscriberCtx == subscriberCtx && subscriber.mCallback == callback) {
            it->second.erase(subscriberIt);
            break;
        }
    }

    if (mEventSubscriberInfoMap[eventVal].size() == 0) {
        mEventSubscriberInfoMap.erase(it);
    }
    else {
        return fmResult;
    }

    switch (eventVal) {
        case LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED:
        case LW2080_NOTIFIERS_LWLINK_ERROR_FATAL:
            if (mFdToGpuUuidMap.size())
                fmResult = unRegisterGpuErrorEvents();
            break;
        case LW000F_NOTIFIERS_FABRIC_EVENT:
            {
                LwRmFreeOsEvent(mRmClientHandle, mFMSessionHandle, mFabricEventsFd);
            }
            break;
    }
    return fmResult;
}


FMIntReturn_t
LocalFMGpuMgr::watchEvents(unsigned int timeout)
{
    //
    // can't take lock here as the GPU detach expects any outstanding poll() loop to exit in order to
    // release all the allocated GPU handles. It will wait for the socketpair fd to be signaled back.
    // Meaning, when detach path write into the socketpair, and this poll() request is outstanding,
    // then it will wake-up poll() immediately and trigger back EventLoopBreakCondition.
    // However, if the poll() call was not outstanding, then this watchEvents() has to execute and
    // issue a poll() request which will immediately trigger back.
    //

    //
    // take a copy/snapshot of the mEventDescriptorList as it may could be modified by event
    // un-registration while walking
    //

    eventDescriptorList_t tempEventDescList = mEventDescriptorList;
    // count the poll break fd also in total fd count
    nfds_t numDescs = tempEventDescList.size() + 1;
    struct pollfd watchers[numDescs];
    unsigned int idx = 0;
    int retVal;
    FMIntReturn_t fmResult;
    eventDescriptorList_t::iterator it;
    for (it = tempEventDescList.begin(); it != tempEventDescList.end(); it++) {
        watchers[idx].fd = *it;
        watchers[idx].events = POLLIN | POLLPRI;
        watchers[idx].revents = 0;
        idx++;
    }

    // always add our socketpair fd to break from poll loop
    watchers[idx].fd = mEventLoopBreakFds[0];
    watchers[idx].events = POLLIN;
    watchers[idx].revents = 0;

#ifdef LW_MODS
    retVal = 0;
#else
    retVal = poll(watchers, numDescs, timeout);
#endif

    if (ShouldStop()) {
        return FM_INT_ST_BREAK;
    }

    if (retVal < 0) {
        return FM_INT_ST_GENERIC_ERROR;
    }

    if (retVal == 0) {
        return FM_INT_ST_OK;
    }

    // check for event loop break is requested
    if (watchers[numDescs-1].revents & POLLIN) {
        // read the socketpair fd to clear its state
        uint16 dummyValue;
        int ret = read(mEventLoopBreakFds[0], &dummyValue,sizeof(dummyValue));
        if (ret == -1) {
            FM_LOG_ERROR("request to read socketpair file descriptor for GPU event loop signaling failed");
        }
        // signal the thread which requested to break the event polling
        lwosCondBroadcast(&m_EventLoopBreakCondition);
        return FM_INT_ST_OK;
    }

    // skip last fd as it is event loop break fd and it is already checked
    for (idx = 0; idx < numDescs-1; idx++) {
        LwUnixEvent rmEvent = {0};
        LwU32 moreEvents = 1;

        // check if there are no events, if so continue
        if (watchers[idx].revents == 0) {
            continue; //no event
        }

        int fd = watchers[idx].fd;
        //there is some event, use rmgeteventdata to get the event
        while (moreEvents) {
            // if the fd corresponding to the fired event is a fabric event, then we need not use
            // LwRmGetEventData.
            if (fd != mFabricEventsFd) {
                LwU32 rmResult;
                rmResult = LwRmGetEventData(mRmClientHandle, watchers[idx].fd, &rmEvent, &moreEvents);
                if (rmResult != LWOS_STATUS_SUCCESS) {
                    // this can happen transiently, mEventDescriptorList is modified from another
                    // thread when this thread is blocked on the poll.
                    // It will be correct at the next poll.
                    FM_LOG_DEBUG("failed to get event data information from GPU Driver, error: %s.",
                                   lwstatusToString(rmResult));
                    return FM_INT_ST_GENERIC_ERROR;
                }
            } else {
                // setting rmEvent.NotifyIndex, to ensure that following code flow
                // remains unchanged
                rmEvent.NotifyIndex = LW000F_NOTIFIERS_FABRIC_EVENT;
		        moreEvents = 0;
            }

            /* Process event and execute the callback for the event */
            /* TODO fill in the errorLinkIndex when event notification
               has the link index for the LWLink error */
            uint32_t errorLinkIndex = 0;
            exelwteEventCallback(fd, rmEvent.NotifyIndex, errorLinkIndex);
        }
    }
    return FM_INT_ST_OK;
}


FMIntReturn_t
LocalFMGpuMgr::exelwteEventCallback(int fd, LwU32 eventValType, uint32_t errorLinkIndex)
{
    fmSubscriberCbArguments_t cbArgs = {0};
    FMUuid_t gpuUuid;

    // take the lock while working on/reading with event fds
    FMAutoLock lock(mLock);

    // for LW000F_NOTIFIERS_FABRIC_EVENT type events the event is not specific to a GPU
    // on the local node but for a GPU on a remote node hence the mFdToGpuUuidMap doesn't
    // contain a valid entry for it. On local node this event is for the FM Session
    if (eventValType != LW000F_NOTIFIERS_FABRIC_EVENT) {
        if (mFdToGpuUuidMap.find(fd) == mFdToGpuUuidMap.end()) {
            FM_LOG_ERROR("event handler is unable to find file descriptor for the specified GPU");
            return FM_INT_ST_GENERIC_ERROR;
        }
        gpuUuid = mFdToGpuUuidMap[fd];
    }
    if (mEventSubscriberInfoMap.find(eventValType) == mEventSubscriberInfoMap.end()) {
        FM_LOG_ERROR("detected event type is not in the subscribed event list");
        return FM_INT_ST_GENERIC_ERROR;
    }
    subscriberInfoList_t subscriberInfoList = mEventSubscriberInfoMap[eventValType];
    subscriberInfoList_t::iterator it;
    gpuEventCallback_t mpCallback;
    switch(eventValType) {
        case LW2080_NOTIFIERS_LWLINK_ERROR_FATAL:
        case LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED:
           {
                //actually not required to do this for all subscribers
                fmGpuErrorInfoCtx_t cbFmGpuErrorArgs = {{{0}}};
                cbFmGpuErrorArgs.gpuError = eventValType;
                cbFmGpuErrorArgs.gpuUuid = gpuUuid;
                cbFmGpuErrorArgs.errorLinkIndex = errorLinkIndex;

                ilwokeEventSubscribers(subscriberInfoList, (void*) &cbFmGpuErrorArgs);
            }
            break;
        //other type of errors can be added
        case LW000F_NOTIFIERS_FABRIC_EVENT:
            {
                //assign callback argument structure and pass to ilwokeEventSubscribers
                void *placeholder = NULL;
                ilwokeEventSubscribers(subscriberInfoList, placeholder);
            }
            break;
    }
    return FM_INT_ST_OK;
}

void
LocalFMGpuMgr::ilwokeEventSubscribers(subscriberInfoList_t subscriberInfoList, void *callbackArgs)
{
    subscriberInfoList_t::iterator it;
    fmSubscriberCbArguments_t cbArgs = {0};
    gpuEventCallback_t mpCallback;
    for(it = subscriberInfoList.begin(); it != subscriberInfoList.end(); it++) {
        subscriberInfo_t mSubscriber = *it;
        cbArgs.subscriberCtx = mSubscriber.subscriberCtx;
        cbArgs.args = callbackArgs;                //arguments required for further function calls from callback
        mpCallback = mSubscriber.mCallback;
        mpCallback((void*) &cbArgs);
    }
}

void
LocalFMGpuMgr::addSubscriberInfoToEventList(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx)
{
    subscriberInfo_t mSubscriberInfo;
    mSubscriberInfo.subscriberCtx = subscriberCtx;
    mSubscriberInfo.mCallback = callback;
    mEventSubscriberInfoMap[eventVal].push_back(mSubscriberInfo);
}

FMIntReturn_t
LocalFMGpuMgr::getAllLwLinkStatus(FMUuid_t gpuUuid, unsigned int &statusMask, unsigned int &activeMask)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS lwLinkStatus = {0};
    LwU32 rmResult;
    LwU32 deviceHandle;
    LwU32 lw2080handle;
    LwU32 linkState;

    if (mInitDoneGpuInfoMap.find(gpuUuid) == mInitDoneGpuInfoMap.end()) {
        FM_LOG_ERROR("request to get LWLink status for an uninitialized GPU UUID: %s", gpuUuid.bytes);
        return FM_INT_ST_GENERIC_ERROR;
    }

    GpuCtxInfo_t gpuCtxInfo = mInitDoneGpuInfoMap[gpuUuid];
    lw2080handle = gpuCtxInfo.lw2080handle;

    activeMask = 0;
    rmResult = LwRmControl(mRmClientHandle, lw2080handle, LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS,
                           &lwLinkStatus, sizeof(lwLinkStatus));
    if(rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("request to get LWLink status for GPU UUID: %s pci bus id: %s failed with error:%s",
                      gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    for(LwU32 i = 0; i < LW2080_CTRL_LWLINK_MAX_LINKS; i++) {
        if(lwLinkStatus.enabledLinkMask & (1U << i)) {
            statusMask |= (1U << i);
            linkState = lwLinkStatus.linkInfo[i].linkState;
            switch (linkState) {
                case LW2080_CTRL_LWLINK_STATUS_LINK_STATE_ACTIVE:
                case LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY:
                case LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY_AC:
                case LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY_RX:
                    activeMask |= (1U << i);
                    break;
                default:
                    break;
            }
        }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuLWLinkSpeedInfo(FMUuid_t gpuUuid, FMLWLinkSpeedInfoList &linkSpeedInfo)
{
    LwU32 rmResult;
    LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS statusParms = {0};

    FM_LOG_DEBUG("getting LWLink speed information for GPU uuid: %s", gpuUuid.bytes);

    // check whether the GPU is initialized before getting speed information
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to get LWLink speed information for an uninitialized GPU uuid: %s",
                        gpuUuid.bytes);
        return FM_INT_ST_BADPARAM;
    }

    // found the specified GPU in our context
    GpuCtxInfo_t &gpuCtxInfo = it->second;

    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle,
                           LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS, &statusParms, sizeof(statusParms));

    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("request to get LWLink speed information for GPU UUID: %s pci bus id: %s failed with error: %s",
                     gpuUuid.bytes, gpuCtxInfo.pciInfo.busId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // parse the desired speed information
    linkSpeedInfo.clear();
    for (LwU32 i = 0; i < LW2080_CTRL_LWLINK_MAX_LINKS; i++) {
        if ((statusParms.enabledLinkMask & (1 << i)) == 0) {
            // skip as the specified link index is not enabled
            continue;
        }

        if (statusParms.linkInfo[i].linkState != LW2080_CTRL_LWLINK_STATUS_LINK_STATE_ACTIVE) {
            // skip speed information for non-active links as it may be garbage and not applicable.
            continue;
        }

        FM_LOG_DEBUG("Link Idx:%d lwlinkLineRateMbps:%d", i, statusParms.linkInfo[i].lwlinkLineRateMbps);
        FM_LOG_DEBUG("Link Idx:%d linkClockMhz:%d", i, statusParms.linkInfo[i].lwlinkLinkClockMhz);
        FM_LOG_DEBUG("Link Idx:%d linkClockType:%d", i, statusParms.linkInfo[i].lwlinkRefClkType);
        FM_LOG_DEBUG("Link Idx:%d linkDataRateKiBps:%d", i, statusParms.linkInfo[i].lwlinkLinkDataRateKiBps);

        // a good active link, copy the speed information from rm.
        FMLWLinkSpeedInfo tempSpeedInfo;
        tempSpeedInfo.linkIndex = i;

        // we are reporting per port bandwidth, so colwert the per lane/link bw accordingly
        if (gpuCtxInfo.archType == LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100) {
            //
            // for GV100, the line rate is 25.78125Gbps and there is 8 lanes (x8)
            // so, the per port peak bandwidth will be (25.78125 * 8)/8 = 25.78125GBps or 25781.25MBps.
            // driver is reporting lwlinkLineRateMbps as 25,781Mbps.
            //
            tempSpeedInfo.linkLineRateMBps = (statusParms.linkInfo[i].lwlinkLineRateMbps * 8)/8;

        }
        else if (gpuCtxInfo.archType == LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100) {
            //
            // for GA100, the line rate is 50Gbps and there is 4 lanes (x4)
            // so, the per port peak bandwidth will be (50 * 4)/8 = 25GBps or 25000MBps.
            // driver is reporting lwlinkLineRateMbps as 50000Mbps.
            //
            tempSpeedInfo.linkLineRateMBps = (statusParms.linkInfo[i].lwlinkLineRateMbps * 4)/8;

            //
            // TODO: Ampere/LimeRock will support optical lanes with clock speed as 53.125Gbps
            // Adjust this computation accordingly when optical support is added.
            //
        }

        tempSpeedInfo.linkClockMhz = statusParms.linkInfo[i].lwlinkLinkClockMhz;
        tempSpeedInfo.linkClockType = statusParms.linkInfo[i].lwlinkRefClkType;
        tempSpeedInfo.linkDataRateKiBps = statusParms.linkInfo[i].lwlinkLinkDataRateKiBps;
        linkSpeedInfo.push_back(tempSpeedInfo);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::setInitDisabledLinkMask(LwU32 gpuId, unsigned int disabledMask)
{
    LwU32 rmResult;
    LW0000_CTRL_GPU_DISABLE_LWLINK_INIT_PARAMS params = { 0 };

    params.gpuId = gpuId;
    params.mask = disabledMask;

    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_GPU_DISABLE_LWLINK_INIT,
                           &params, sizeof(params));

    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_ERROR("failed to set LWLink initialization disabled mask for probed GPU ID: %d, with error : %s",
                     gpuId, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::getGfid(FMUuid_t gpuUuid, FMPciInfo_t &vf, uint32_t &gfid, uint32_t &gfidMask)
{
    LW2080_CTRL_GPU_GET_GFID_PARAMS params = { 0 };
    LwU32 rmResult;

    FM_LOG_DEBUG("getGfid: GPU %s VF 0x%x.%x.%x.%x", gpuUuid.bytes, vf.domain, vf.bus, vf.device, vf.function);

    // check whether the GPU is initialized before getting its GFID.
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to get gfid for an uninitialized GPU, uuid: %s count of GPUs %d",
                       gpuUuid.bytes, (int)mInitDoneGpuInfoMap.size() );
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;

    params.domain = vf.domain;
    params.bus = vf.bus;
    params.device = vf.device;
    params.func = vf.function;

    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle, LW2080_CTRL_CMD_GPU_GET_GFID, &params, sizeof(params));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("request to get gfid for GPU uuid: %s failed with error %s", gpuUuid.bytes, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    gfid = params.gfid;
    gfidMask = params.gfidMask;

    FM_LOG_DEBUG("getGfid: GPU %s gfid %x mask %x", gpuUuid.bytes, gfid, gfidMask);

    return FM_INT_ST_OK;
}

//
// configGfid() is used to set or clear GPU's GFID in the RM driver based on the activate flag.
//
// Set GFID - while activating vGPU GPU partition
// Clear GFID - while deactivating vGPU GPU partition
//
FMIntReturn_t
LocalFMGpuMgr::configGfid(FMUuid_t gpuUuid, uint32_t gfid, bool activate)
{
    LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY_PARAMS params = { 0 };
    LwU32 rmResult;

    FM_LOG_DEBUG("configGfid: GPU %s gfid %x activate %d", gpuUuid.bytes, gfid, activate);

    // check whether the GPU is initialized before getting its GFID.
    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(gpuUuid);
    if (it == mInitDoneGpuInfoMap.end()) {
        FM_LOG_WARNING("trying to configure gfid for an uninitialized GPU, uuid: %s count of GPUs %d",
                       gpuUuid.bytes, (int)mInitDoneGpuInfoMap.size());
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t &gpuCtxInfo = it->second;

    params.gfid = gfid;
    params.bEnable = activate;

    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle, LW2080_CTRL_CMD_GPU_UPDATE_GFID_P2P_CAPABILITY,
                           &params, sizeof(params));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("request to configure gfid for GPU uuid: %s failed with error %s",
                     gpuUuid.bytes, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

bool
LocalFMGpuMgr::isGpuArchitectureSupported(GpuCtxInfo_t &gpuCtxInfo)
{
    LW2080_CTRL_MC_GET_ARCH_INFO_PARAMS lw2080GetArchInfoParams = {0};
    LwU32 rmResult;

    FM_LOG_DEBUG("getting architecture type for GPU pci bus id: %s", gpuCtxInfo.pciInfo.busId);

    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle, LW2080_CTRL_CMD_MC_GET_ARCH_INFO,
                           &lw2080GetArchInfoParams, sizeof(lw2080GetArchInfoParams));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to get architecture type for GPU pci bus id: %s with error:%s",
                     gpuCtxInfo.pciInfo.busId,  lwstatusToString(rmResult));
        // treat error as not supported
        return false;
    }

    FM_LOG_DEBUG("GPU pci bus id: %s architecture type is: %x", gpuCtxInfo.pciInfo.busId,
                 lw2080GetArchInfoParams.architecture);

    gpuCtxInfo.archType = lw2080GetArchInfoParams.architecture;
    // got architecture type from driver
    switch (gpuCtxInfo.archType) {
         // these gpu architecture types are supported
        case LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100:
        case LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GH100:
#endif

            return true;
        default:
            // not supported architecture
            FM_LOG_WARNING("not opening/attaching GPU pci bus id: %s as its architecture type is not supported",
                           gpuCtxInfo.pciInfo.busId);
            return false;
    }

    // default, not supported
    return false;
}

FMIntReturn_t
LocalFMGpuMgr::getGpuLWLinkCapInfo(GpuCtxInfo_t &gpuCtxInfo)
{
    LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS_PARAMS capsParms = {0};
    LwU32 rmResult;

    FM_LOG_DEBUG("getting LWLink capabilities for GPU pci bus id: %s", gpuCtxInfo.pciInfo.busId);

    rmResult = LwRmControl(mRmClientHandle, gpuCtxInfo.lw2080handle, LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS,
                           &capsParms, sizeof(capsParms));

    if (rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("failed to get LWLink capabilities for GPU pci bus id: %s with error:%s",
                     gpuCtxInfo.pciInfo.busId,  lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }

    // save the discovered and enabled link mask information
    gpuCtxInfo.discoveredLinkMask = capsParms.discoveredLinkMask;
    gpuCtxInfo.enabledLinkMask = capsParms.enabledLinkMask;

    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::registerEventForGpu(FMUuid_t &uuidInfo)
{
    FMIntReturn_t fmResult;

    //
    // Note: this function is called from the context of initializeGpu(), which
    // holds the necessary locks for event registration
    //

    InitDoneGpuInfoMap_t::iterator it = mInitDoneGpuInfoMap.find(uuidInfo);
    if ( it == mInitDoneGpuInfoMap.end() ) {
        FM_LOG_ERROR("GPU %s is not initialized.", uuidInfo.bytes);
        return FM_INT_ST_BADPARAM;
    }

    // First allocate OS event for the GPU
    GpuCtxInfo_t &gpuCtxInfo = it->second;
    LwU32 lw2080handle = gpuCtxInfo.lw2080handle;
    LwU32 gpuIndex = gpuCtxInfo.gpuId;

    int fd;
    LwU32 rmResult = LwRmAllocOsEvent(mRmClientHandle, lw2080handle, NULL, &fd);
    if(rmResult != LWOS_STATUS_SUCCESS) {
        FM_LOG_ERROR("request to allocate GPU Driver event monitoring object for probed GPU ID: %d failed with error: %s",
                     (int) gpuIndex, lwstatusToString(rmResult));
        return FM_INT_ST_GENERIC_ERROR;
    }
    mFdToGpuUuidMap[fd] = uuidInfo;
    mEventDescriptorList.push_back(fd);

    // Allocate GPU error events
    fmResult = allocEventForGpuErrorType(LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED, fd, lw2080handle);
    if (fmResult != FM_INT_ST_OK) {
        return fmResult;
    }

    fmResult = allocEventForGpuErrorType(LW2080_NOTIFIERS_LWLINK_ERROR_FATAL, fd, lw2080handle);
    if (fmResult != FM_INT_ST_OK) {
        return fmResult;
    }

    //
    // TODO: signal the GPU event watch poll() loop to do the GPU event monitoring immediately instead of
    // waiting for the poll() loop to timeout and start watching in next iteration (10 seconds window for now)
    //
    return FM_INT_ST_OK;
}

FMIntReturn_t
LocalFMGpuMgr::unRegisterEventForGpu(FMUuid_t &uuidInfo)
{
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    //
    // Note: this function is called from the context of deInitializeGpu(), which
    // holds the necessary locks for event registration
    //

    if (mInitDoneGpuInfoMap.find(uuidInfo) == mInitDoneGpuInfoMap.end()) {
        FM_LOG_ERROR("GPU %s is not initialized.", uuidInfo.bytes);
        return FM_INT_ST_BADPARAM;
    }

    GpuCtxInfo_t gpuCtxInfo = mInitDoneGpuInfoMap[uuidInfo];
    LwU32 lw2080handle = gpuCtxInfo.lw2080handle;
    LwU32 gpuIndex = gpuCtxInfo.gpuId;
    int fd = -1;

    fileDescriptorToGpuUuidMap_t::iterator it;
    for (it = mFdToGpuUuidMap.begin(); it != mFdToGpuUuidMap.end(); it++ ) {
        fd = it->first;
        FMUuid_t gpuUuid = it->second;

        if (strncmp(gpuUuid.bytes, uuidInfo.bytes, FM_UUID_BUFFER_SIZE) != 0) {
            continue;
        }

        fmResult = freeOsEvent(fd, lw2080handle);
        if (fmResult != FM_INT_ST_OK) {
            // error already logged
            return fmResult;
        }
        break;
    }

    if (it != mFdToGpuUuidMap.end()) {
        mFdToGpuUuidMap.erase(it);
    }

    if (fd >= 0) {
        eventDescriptorList_t::iterator fdIt = mEventDescriptorList.begin();

        while (fdIt != mEventDescriptorList.end()) {
            if (*fdIt == fd) {
                fdIt = mEventDescriptorList.erase(fdIt);
                continue;
            }
            fdIt++;
        }
    }

    // write to the sockerpair fd to wake-up/break the GPU event watch poll() loop
    uint16 dummyValue = 1;
    int ret = write(mEventLoopBreakFds[1], &dummyValue, sizeof(dummyValue));

    if (ret == -1) {
        FM_LOG_ERROR("request to write to file descriptor in order to wake-up GPU event watching thread failed");
    }
    //
    // wait for the event loop to signal us back. timeout is not treated as an error
    // as RM will eventually wait for all handles to be closed.
    //
    int condStatus = lwosCondWait(&m_EventLoopBreakCondition, &m_EventLoopBreakMutex, 2000);
    if (condStatus == LWOS_TIMEOUT) {
        FM_LOG_WARNING("request to exit GPU event monitoring timed out. GPU may have outstanding handles");
    }

    return fmResult;
}
