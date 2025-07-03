/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
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
#include <list>
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "FmThread.h"
#include "LocalFabricManager.h"


typedef struct  {
    FMUuid_t gpuUuid;
    uint64_t errorLinkIndex;
    int gpuError;
} fmGpuErrorInfoCtx_t;

typedef struct  {
    void *subscriberCtx;
    void *args;
} fmSubscriberCbArguments_t;
 
typedef void (*gpuEventCallback_t)(void*);
 
class LocalFMGpuMgr : public FmThread
{
 
public:
 
    LocalFMGpuMgr();
    ~LocalFMGpuMgr();
 
    virtual void run();
 
    FMIntReturn_t initializeAllGpus();
    FMIntReturn_t deInitializeAllGpus();
    FMIntReturn_t initializeGpu(FMUuid_t &uuidToInit, bool registerEvt);
    FMIntReturn_t deInitializeGpu(FMUuid_t &uuidToDeinit, bool unRegisterEvt);
    FMIntReturn_t getInitializedGpuInfo(FMGpuInfoList &initDoneGpuInfo);
    FMIntReturn_t getExcludedGpuInfo(std::list<FMExcludedGpuInfo_t> &excludedGpuInfo);
    FMIntReturn_t allocFmSession(unsigned int abortLwdaJobsOnFmExit);
    FMIntReturn_t freeFmSession();
    FMIntReturn_t fmSessionSetState();
    FMIntReturn_t fmSessionSetNodeId(uint32 nodeId);
    FMIntReturn_t fmSessionClearState();
    FMIntReturn_t setGpuFabricGPA(FMUuid_t uuid, unsigned long long baseAddress);
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    FMIntReturn_t setGpuFabricGPAEgm(FMUuid_t uuid, unsigned long long egmBaseAddress);
#endif
    FMIntReturn_t setGpuFabricFLA(FMUuid_t uuid, unsigned long long baseAddress, unsigned long long size);
    FMIntReturn_t clearGpuFabricFLA(FMUuid_t uuid, unsigned long long baseAddress, unsigned long long size);
    FMIntReturn_t subscribeForGpuEvents(LwU32 eventVal, gpuEventCallback_t, void *subscriberCtx);
    FMIntReturn_t unsubscribeGpuEvents(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx);
    FMIntReturn_t startGpuEventWatch();
    void stopGpuEventWatch();
    FMIntReturn_t getAllLwLinkStatus(FMUuid_t gpuUuid, unsigned int &statusMask, unsigned int &activeMask);
    FMIntReturn_t getGpuLWLinkSpeedInfo(FMUuid_t gpuUuid, FMLWLinkSpeedInfoList &linkSpeedInfo);
    FMIntReturn_t setGpuLWLinkInitDisabledMask(FMUuid_t uuid, unsigned int disabledMask);
    FMIntReturn_t refreshRmLibProbedGpuInfo();
    FMIntReturn_t getGfid(FMUuid_t gpuUuid, FMPciInfo_t &vf, uint32_t &gfid, uint32_t &gfidMask);
    FMIntReturn_t configGfid(FMUuid_t gpuUuid, uint32_t gfid, bool activate);

    LwHandle getMemSubDeviceHandleForGpuId(LwU32 gpuId);
    LwHandle getRmClientHandle();
    LwHandle getFmSessionHandle();
    
    LwU32 getProbedGpuCount();
    

private:
    typedef struct {
        unsigned int gpuId; // for RM communication
        FMPciInfo_t   pciInfo;
        LwU32       deviceInstance; // this gpuIndex
        LwU32       subDeviceInstance;
        LwU32       archType;
        LwU32       discoveredLinkMask;
        LwU32       enabledLinkMask;
        LwU32       gpuInstance;
        LwU32       lw0080handle;
        LwU32       lw2080handle;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        bool        isEgmCapable;
        bool        isSpaCapable;
        LwU64       spaAddress;
#endif
    } GpuCtxInfo_t;

    typedef struct {
        void *subscriberCtx;
        void (*mCallback)(void*);
    } subscriberInfo_t;
 
    struct UUIDComparer {
        bool operator()(const FMUuid_t& left, const FMUuid_t& right) const
        {
            return strncmp(left.bytes, right.bytes, sizeof(FMUuid_t)) < 0;
        }
    };
 
    typedef std::map<FMUuid_t, GpuCtxInfo_t, UUIDComparer> InitDoneGpuInfoMap_t;

    //data structures required for subscribe
    typedef std::list<int> eventDescriptorList_t;                               // List of File descriptors to watch
    typedef std::map<int, FMUuid_t> fileDescriptorToGpuUuidMap_t;               // Map of Gpu UUID and Fd -> for gpu lw link errors
    typedef std::list<subscriberInfo_t> subscriberInfoList_t;                   // Subscriber info list required for callback
    typedef std::map<LwU32, subscriberInfoList_t> eventSubscriberInfoMap_t;     // Mapping between event and subscriber info list
 
    fileDescriptorToGpuUuidMap_t mFdToGpuUuidMap;
    eventDescriptorList_t mEventDescriptorList;
    
    eventSubscriberInfoMap_t mEventSubscriberInfoMap;

    int mEventLoopBreakFds[2];
    LWOSCriticalSection m_EventLoopBreakMutex;
    lwosCV m_EventLoopBreakCondition;
    int mFabricEventsFd;

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
    FMIntReturn_t registerEvent(LwU32 eventVal);
    FMIntReturn_t unRegisterEvent(LwU32 eventVal, gpuEventCallback_t callback, void *subscriberCtx);
    FMIntReturn_t unRegisterGpuErrorEvents();
    FMIntReturn_t allocOsEventForGpus();
    FMIntReturn_t freeOsEvent(int fd, LwU32 lw2080handle);
    FMIntReturn_t allocEventForGpuErrorType(LwU32 eventVal, int &fd, LwU32 lw2080handle);
    FMIntReturn_t allocRmEvent();
    FMIntReturn_t watchEvents(unsigned int timeout);
    void addSubscriberInfoToEventList(LwU32 eventVal, gpuEventCallback_t, void *subscriberCtx);
    void ilwokeEventSubscribers(subscriberInfoList_t subscriberInfoList, void *callbackArgs);
    FMIntReturn_t exelwteEventCallback(int fd, LwU32 rmEvent, uint32_t errorLinkIndex);
    FMIntReturn_t setInitDisabledLinkMask(LwU32 gpuId, unsigned int disabledMask);

    bool isGpuArchitectureSupported(GpuCtxInfo_t &gpuCtxInfo);
    FMIntReturn_t getGpuLWLinkCapInfo(GpuCtxInfo_t &gpuCtxInfo);

    FMIntReturn_t registerEventForGpu(FMUuid_t &uuidInfo);
    FMIntReturn_t unRegisterEventForGpu(FMUuid_t &uuidInfo);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    FMIntReturn_t getGpuCapabilityInfo(GpuCtxInfo_t &gpuCtxInfo);
#endif

    LwHandle mRmClientHandle;
    LwHandle mFMSessionHandle;
    LwHandle mFabricEventsHandle;
    LWOSCriticalSection mLock;
    InitDoneGpuInfoMap_t mInitDoneGpuInfoMap;
};

