/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
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
#include "fabricmanager.pb.h"
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include "FMCommCtrl.h"
#include "FMLWLinkTypes.h"
#include "FMLWLinkError.h"
#include "lwos.h"
#include <g_lwconfig.h>

/*****************************************************************************/
/*  Fabric Manager Link Training related interfaces                          */
/*****************************************************************************/

/*
 * Provides interfaces for exchanging LWLink related messages to Local FM 
 * such as link initialization, discovery request, train connection etc.
 * This class is used only by Global FM and abstract all the LWLink related
 * functionality for upper layers.
 * 
 * The caller fills appropriate member of FMLWLinkReq based on the intended
 * LWLink request and the corresponding result is returned in FMLWLinkReqResult
 * The requests are asynchronous in nature, means the requests will be returned
 * immediately with a unique requestId (handle) and the caller can check back the
 * status of the request and result using the isLinkReqComplete() method.
 * 
 * Dependency: FMConnInterface - provides methods for sending messages to peer LFM.
 * 
 * Note :  Can't make the members in FMLWLinkReq/FMLWLinkReqResult as union, 
 * as the individual structures have STL types, which are not POD types.
 */

// node init will be used for data enable, rx cal, set common mode and link init.
typedef struct
{
    uint32 nodeId;
} FMLWLinkNodeInitReq;

typedef struct
{
    // nothing specific other than the status, which is part of FMLWLinkReqResult
} FMLWLinkNodeInitResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkNodeInitStatusReq;

typedef struct
{
    FMLinkInitStatusInfoList statusInfo;
} FMLWLinkNodeInitStatusResp;

typedef struct
{
    uint32 nodeId;
    uint64 switchId;
    uint64 linkMask;
} FMLWLinkNodeInitResetSwitchLinksReq;

typedef struct
{
    // nothing specific other than the status, which is part of FMLWLinkReqResult
} FMLWLinkNodeInitResetSwitchLinksResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkNodeInitResetAllSwitchLinksReq;

typedef struct
{
    // nothing specific other than the status, which is part of FMLWLinkReqResult
} FMLWLinkNodeInitResetAllSwitchLinksResp;

typedef struct
{
    uint32 masterNodeId;
    uint64 masterGpuOrSwitchId;
    uint32 masterLinkIndex;
    uint32 slaveNodeId;
    uint64 slaveGpuOrSwitchId;
    uint32 slaveLinkIndex;
    FMLWLinkTrainType trainTo;
} FMLWLinkConnTrainReq;

typedef struct
{
    FMLWLinkStateInfo masterState;
    FMLWLinkStateInfo slaveState;
} FMLWLinkConnTrainResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkDiscoverIntraNodeConnReq;

typedef struct
{
    // nothing specific other than the status
} FMLWLinkDiscoverIntraNodeConnResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkWriteDiscoveryTokenReq;

typedef struct
{
    FMLWLinkDiscoveryTokenList tokenInfo;
} FMLWLinkWriteDiscoveryTokenResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkReadDiscoveryTokenReq;

typedef struct
{
    uint32 nodeId;
} FMLWLinkReadSidReq;

typedef struct
{
    FMLWLinkDiscoveryTokenList tokenInfo;
} FMLWLinkReadDiscoveryTokenResp;

typedef struct
{
    FMLWLinkSidList sidList;
} FMLWLinkReadLinkSidResp;

typedef struct
{
    FMLWLinkEndPointInfo       localEndInfo;
    FMLWLinkRemoteEndPointInfo remoteEndInfo;
} FMLWLinkAddInterNodeConnReq;

typedef struct
{
    // nothing specific other than the status
} FMLWLinkAddInterNodeConnResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkGetIntraNodeConnReq;

typedef struct
{
    FMLWLinkConnList connInfo;
} FMLWLinkGetIntraNodeConnResp;

typedef struct
{
    uint32 nodeId;
} FMLWLinkGetDeviceLwlinkStateReq;

typedef struct
{
    FMLWLinkEndPointInfo lwEndInfo;
    FMLWLinkStateInfo    stateInfo;
} FMLWLinkGetDeviceLwlinkStateRespDetailed;

typedef std::vector<FMLWLinkGetDeviceLwlinkStateRespDetailed> FMLWLinkGetDeviceLwlinkStateResp;

typedef std::vector<FMLWLinkConnTrainReq>       FMLWLinkConnTrainParallelReq;


typedef struct
{
    uint32 masterNodeId;
    uint64 masterGpuOrSwitchId;
    uint32 masterLinkIndex;
    uint32 slaveNodeId;
    uint64 slaveGpuOrSwitchId;
    uint32 slaveLinkIndex;
    FMLWLinkStateInfo masterState;
    FMLWLinkStateInfo slaveState;
    FMLWLinkQualityInfo masterQualityInfo;
    FMLWLinkQualityInfo slaveQualityInfo;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FMLWLinkFomValues fomValues;
    FMLWLinkGradingValues gradingValues;
#endif
} FMLWLinkConnTrainRespDetailed;

typedef std::vector<FMLWLinkConnTrainRespDetailed>      FMLWLinkConnTrainParallelResp;

typedef struct
{
    uint32 nodeId;
    uint64 switchId;
    uint64 attemptedMask0;
    uint64 failedMask0;    
} FMLWLinkSwitchTrainingFailedLinkInfoReq;

typedef struct
{
    // nothing specific other than the status, which is part of FMLWLinkReqResult
} FMLWLinkSwitchTrainingFailedLinkInfoResp;
typedef std::vector<FMLWLinkEndPointInfo>      FMLWLinkEndPointInfoList;

typedef struct 
{
    FMLWLinkNodeInitReq                         nodeInitReq;
    FMLWLinkNodeInitStatusReq                   nodeInitStatusReq;
    FMLWLinkNodeInitResetSwitchLinksReq         nodeInitResetSwitchLinksReq;
    FMLWLinkNodeInitResetAllSwitchLinksReq      nodeInitResetAllSwitchLinksReq;
    FMLWLinkConnTrainReq                        connTrainReq;
    FMLWLinkConnTrainParallelReq                connTrainParallelReq;
    FMLWLinkDiscoverIntraNodeConnReq            discoverIntraNodeConnReq;
    FMLWLinkReadDiscoveryTokenReq               readDiscoveryTokenReq;
    FMLWLinkReadSidReq                          readSidReq;
    FMLWLinkWriteDiscoveryTokenReq              writeDiscoveryTokenReq;
    FMLWLinkAddInterNodeConnReq                 addInterNodeConnReq;
    FMLWLinkGetIntraNodeConnReq                 getIntraNodeConnReq;
    FMLWLinkSwitchTrainingFailedLinkInfoReq     switchTrainingFailedLinkInfoReq;
    FMLWLinkGetDeviceLwlinkStateReq             getDeviceLwlinkStateReq;
} FMLWLinkReq;

typedef struct 
{
    uint64                                      requestId;
    LWLinkErrorCodes                            status;
    FMLWLinkNodeInitResp                        nodeInitResp;
    FMLWLinkNodeInitStatusResp                  nodeInitStatusResp;
    FMLWLinkNodeInitResetSwitchLinksResp        nodeInitResetLinkResp;
    FMLWLinkNodeInitResetAllSwitchLinksResp     nodeInitResetAllSwitchLinksResp;
    FMLWLinkConnTrainResp                       connTrainResp;
    FMLWLinkConnTrainParallelResp               connTrainParallelResp;
    FMLWLinkDiscoverIntraNodeConnResp           discoverIntraNodeConnResp;
    FMLWLinkReadDiscoveryTokenResp              readDiscoveryTokenResp;
    FMLWLinkReadLinkSidResp                     readSidResp;
    FMLWLinkWriteDiscoveryTokenResp             writeDiscoveryTokenResp;
    FMLWLinkAddInterNodeConnResp                addInterNodeConnResp;
    FMLWLinkGetIntraNodeConnResp                getIntraNodeConnResp;
    FMLWLinkSwitchTrainingFailedLinkInfoResp    switchTrainingFailedLinkInfoResp;
    FMLWLinkGetDeviceLwlinkStateResp            getDeviceLwlinkStateResp;
} FMLWLinkReqResult;

class GlobalFMLWLinkIntf : public FMMessageHandler
{
public:
    GlobalFMLWLinkIntf(FMConnInterface *ctrlConnIntf);

    virtual ~GlobalFMLWLinkIntf();

    FMIntReturn_t sendTrainRequest(lwswitch::FabricManagerMessageType msgType,
                                   FMLWLinkReq &linkReq,
                                   uint64 &requestId);

    FMIntReturn_t sendResetSwitchLinksReq(FMLWLinkReq &linkReq,
                                          uint64 &requestId);

    FMIntReturn_t sendResetAllSwitchLinksReq(FMLWLinkReq &linkReq,
                                             uint64 &requestId);

    FMIntReturn_t sendConnTrainReq(FMLWLinkReq &linkReq,
                                   uint64 &requestId);

    FMIntReturn_t sendConnTrainParallelReq(FMLWLinkReq &linkReq,
                                           uint64 &requestId);

    FMIntReturn_t sendGetDeviceLwlinkStateReq(FMLWLinkReq &linkReq,
                                              uint64 &requestId);

    FMIntReturn_t sendDiscoverIntraNodeConnReq(FMLWLinkReq &linkReq,
                                               uint64 &requestId);

    FMIntReturn_t sendAddInterNodeConnReq(FMLWLinkReq &linkReq,
                                          uint64 &requestId);

    FMIntReturn_t sendGetIntraNodeConnReq(FMLWLinkReq &linkReq,
                                          uint64 &requestId);

    FMIntReturn_t sendWriteDiscoveryReq(FMLWLinkReq &linkReq,
                                        uint64 &requestId);

    FMIntReturn_t sendReadSidReq(FMLWLinkReq &linkReq,
                                 uint64 &requestId);

    FMIntReturn_t sendReadDiscoveryReq(FMLWLinkReq &linkReq,
                                       uint64 &requestId);

    FMIntReturn_t sendSwitchTrainingFailedLinkInfo(FMLWLinkReq &linkReq,
                                                   uint64 &requestId);

    bool isLinkReqComplete(uint64 requestId,
                           FMLWLinkReqResult &reqResult);

    // implementation of virtual methods for message and event handling

    // this class handles only the response of the link requests send out
    // ie, FM_LWLINK_TRAIN_RSP_COMPLETE message.
    virtual void handleMessage(lwswitch::fmMessage *pFmMessage);

    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    // maintain book keeping information of outstanding LWLink requests
    typedef struct
    {
        uint32 toNodeId;
        lwswitch::FabricManagerMessageType reqType;
        FMLWLinkReq req;
        FMLWLinkReqResult result;
    } FMLWLinkReqCtx;

    void genConnTrainParallelReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                    FMLWLinkReq &linkReq);

    void genConnTrainReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                            FMLWLinkReq &linkReq);

    void genNodeInitReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                           FMLWLinkReq &linkReq);

    void genNodeInitStatusReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                 FMLWLinkReq &linkReq);

    void genNodeInitResetSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                           FMLWLinkReq &linkReq);

    void genNodeInitResetAllSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                              FMLWLinkReq &linkReq);

    void genDiscoverIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                        FMLWLinkReq &linkReq);

    void genAddInterNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                   FMLWLinkReq &linkReq);

    void genGetIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                   FMLWLinkReq &linkReq);

    void genWriteDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                 FMLWLinkReq &linkReq);

    void genReadSidReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                          FMLWLinkReq &linkReq);

    void genGetDeviceLwlinkStateReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                       FMLWLinkReq &linkReq);

    void genReadDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                FMLWLinkReq &linkReq);

    void genSwitchTrainingFailedReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                       FMLWLinkReq &linkReq);

    void addToReqPendingTable(uint64 reqId,
                              FMLWLinkReq &linkReq,
                              lwswitch::FabricManagerMessageType reqType,
                              uint32 toNodeId);

    void removeFromReqPendingTable(uint64 reqId);

    bool getPendingTrainReq(uint64 reqId, FMLWLinkReqCtx &reqCtx);

    void markPendingReqAsComplete(uint64 reqId, FMLWLinkReqCtx &reqCtx);

    uint64 getNextTrainReqId(void);

    void handleTrainReqResponseMsg(lwswitch::fmMessage * pFmMessage);

    void handleNodeDisconnect(uint32 nodeId);

    void parseConnTrainParallelResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                    FMLWLinkReqCtx &reqCtx);

    void parseConnTrainResp(lwswitch::lwlinkResponseMsg &rspMsg,
                            FMLWLinkReqCtx &reqCtx);

    void parseWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                      FMLWLinkReqCtx &reqCtx);

    void parseReadSidResp(lwswitch::lwlinkResponseMsg &rspMsg,
                          FMLWLinkReqCtx &reqCtx);

    void parseGetDeviceLwlinkStateResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                       FMLWLinkReqCtx &reqCtx);

    void parseReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                     FMLWLinkReqCtx &reqCtx);

    void parseLinkInitStatusResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                 FMLWLinkReqCtx &reqCtx);

    void parseGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                   FMLWLinkReqCtx &reqCtx);

    // debug function helpers

    void dumpTrainCtxEntry(std::ostream *os,
                           uint64 reqId,
                           FMLWLinkReqCtx &reqCtx);

    // class members
    typedef std::map <uint64, FMLWLinkReqCtx> TrainRequestMap;
    TrainRequestMap  mTrainReqPending;  // outstanding requests
    TrainRequestMap  mTrainReqComplete; // completed requests

    // the maps needs to be protected as it is changed from two thread contexts
    LWOSCriticalSection mLock;

    FMConnInterface *mCtrlConnIntf;
    uint64 mNextReqId;
};
