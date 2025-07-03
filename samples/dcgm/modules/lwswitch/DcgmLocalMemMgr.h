#pragma once

#include "LwcmThread.h"
#include "fabricmanager.pb.h"
//TODO:including DcgmFMCommon.h to get definition of uint32. Where is a better palce to get this 
//from for FM in general
#include "DcgmFMCommon.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl000f.h"
#include "lwos.h"


class DcgmLocalFabricManagerControl;
class DcgmFMLocalCoOpMgr;

#define FM_EVENTS_POLL_INTERVAL_MS    30000

//These two will not be needed once RM events API is checked in
#define FM_EVENTS_DEV_PATH  "/tmp/rm_events"
#define FM_EVENTS_POLL_NUM_FDS     1

//These two will be provided by RM when APIs get checked in
#define LW000F_CTRL_FABRIC_EVENT_TYPE_IMPORT  0
#define LW000F_CTRL_FABRIC_EVENT_TYPE_RELEASE 1

class DcgmLocalMemMgr : public LwcmThread
{
public:
    DcgmLocalMemMgr(DcgmFMLocalCoOpMgr *pLocalCoopMgr, DcgmLocalFabricManagerControl *pLocalFmControl);
    ~DcgmLocalMemMgr();

    virtual void run(void);
    void handleMessage(lwswitch::fmMessage *pFmMessage);
private:
    bool processEvents();
    bool sendImportRequest(LW000F_CTRL_FABRIC_EVENT &eventData);
    bool sendImportResponse(const lwswitch::memoryImportReq &reqMsg, uint32 peerNodeId, uint32 errCode);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool sendImportError(const lwswitch::memoryImportRsp &rspMsg, uint32 peerNodeId, uint32 errCode);
#endif
    bool sendUnimportRequest(LW000F_CTRL_FABRIC_EVENT &eventData);
    bool sendUnimportResponse(const lwswitch::memoryUnimportReq &reqMsg, uint32 peerNodeId, uint32 errCode);
    bool waitForEvents();
    bool allocRmEvent();
    bool initMemMgr();

    bool probeDevices(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams);
    bool getPciInfo(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams, LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS *pciInfoParams);
    bool attachGpus(const LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS *pciInfoParams);
    bool allocDevices();
    LwHandle getDeviceHandleForGpu(LwU32 gpuIndex);
    LwHandle getHandleForFmSession();
    void createProbedGpuIdToIndexMap(LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS &probeParams);
    LwU32 getProbedGpuIdToIndex(LwU32 gpuId);

    LwHandle getMemEventsHandle();
    LwHandle getMemSubDeviceHandleForGpu(LwU32 gpuIndex);

    bool handleImportRequest(lwswitch::fmMessage *pFmMessage);
    bool handleImportResponse(lwswitch::fmMessage *pFmMessage);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool handleImportError(lwswitch::fmMessage *pFmMessage);
    bool readPageTableEntries(lwswitch::memoryImportRsp *rspMsg, LwHandle objectHandle);
#endif
    bool handleUnimportRequest(lwswitch::fmMessage *pFmMessage);
    bool handleUnimportResponse(lwswitch::fmMessage *pFmMessage);

    DcgmFMLocalCoOpMgr *mFMLocalCoOpMgr;
    DcgmLocalFabricManagerControl *mLocalFabricManagerControl;

    LwHandle mHandleFmClient;
    LwHandle mHandleFmSession;
    LwHandle mHandleOsEvent;
    LwU32 mDeviceCount;
    //This map is used to get the GPU index from the GPU IDs in import/export events
    std::map<LwU32, LwU32> mGpuProbedGpuIdToIndexMap;

    
    //TODO temporary till RM events are accessible via RM events interface
    int mRMFd;
    LWOSCriticalSection mLock;
};
