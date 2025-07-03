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

#pragma once

#include <map>
#include <vector>

#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"

#include "lwtypes.h"

class LocalFMGpuMgr
{
public:

    LocalFMGpuMgr();
    ~LocalFMGpuMgr();

    FMIntReturn_t initializeAllGpus();
    FMIntReturn_t deInitializeAllGpus();

    FMIntReturn_t initializeGpu(FMUuid_t &uuidToInit);
    FMIntReturn_t deInitializeGpu(FMUuid_t &uuidToDeinit);

    FMIntReturn_t getInitializedGpuInfo(std::vector<FMGpuInfo_t> &initDoneGpuInfo);
    FMIntReturn_t getExcludedGpuInfo(std::vector<FMExcludedGpuInfo_t> &excludedGpuInfo);

    FMIntReturn_t allocFmSession();
    FMIntReturn_t freeFmSession();
    FMIntReturn_t fmSessionSetState();
    FMIntReturn_t fmSessionClearState();

    FMIntReturn_t setGpuFabricBaseAddress(FMUuid_t gpuUuid, unsigned long long baseAddress);

private:

    typedef struct {
        unsigned int gpuId; // for RM communication
        FMPciInfo_t pciInfo;
        LwU32       deviceInstance; // this gpuIndex
    LwU32       archType;
        LwU32       subDeviceInstance;
    LwU32       discoveredLinkMask;
        LwU32       enabledLinkMask;
        LwU32       gpuInstance;
        LwU32       lw0080handle;
        LwU32       lw2080handle;
    } GpuCtxInfo_t;

    struct UUIDComparer {
        bool operator()(const FMUuid_t& left, const FMUuid_t& right) const
        {
            return memcmp(left.bytes, right.bytes, sizeof(FMUuid_t)) < 0;
        }
    };

    typedef std::map<FMUuid_t, GpuCtxInfo_t, UUIDComparer> InitDoneGpuInfoMap_t;
    InitDoneGpuInfoMap_t mInitDoneGpuInfoMap;
    
    FMIntReturn_t getGpuLWLinkCapInfo(GpuCtxInfo_t &gpuCtxInfo);
    bool isGpuArchitectureSupported(GpuCtxInfo_t &gpuCtxInfo);
    FMIntReturn_t getProbedGpuCount(std::vector<LwU32> &gpuIds);
    FMIntReturn_t attachAndOpenGpuHandles(LwU32 gpuId, FMPciInfo_t &pciInfo, FMUuid_t &uuid);
    FMIntReturn_t detachAndCloseGpuHandles(GpuCtxInfo_t &gpuCtxInfo, FMUuid_t &uuid);
    FMIntReturn_t getGpuPciInfo(LwU32 gpuId, FMPciInfo_t &pciInfo);
    FMIntReturn_t getGpuUuidInfo(LwU32 gpuId, FMUuid_t &uuidInfo);
    bool isGpuInDrainState(LwU32 gpuId);
    FMIntReturn_t doRmAttachGpu(LwU32 gpuId);
    FMIntReturn_t doRmDeatchGpu(LwU32 gpuId);
    FMIntReturn_t getGpuRmDevInstanceIdInfo(GpuCtxInfo_t &gpuCtxInfo);
    FMIntReturn_t allocateGpuSubDevices(GpuCtxInfo_t &gpuCtxInfo);
    FMIntReturn_t freeGpuSubDevices(GpuCtxInfo_t &gpuCtxInfo);
    LwU32 getDeviceHandleForGpu(LwU32 gpuIndex);
    LwU32 getMemSubDeviceHandleForGpu(LwU32 gpuIndex);

    static const LwU32 mGpuDevHandleBase;
    static const LwU32 mGpuSubDevHandleBase;

    LwU32 mRmClientFd;
    LwU32 mFMSessionFd;
};

