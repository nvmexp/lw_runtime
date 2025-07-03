#pragma once

#include <map>
#include "fabricmanager.pb.h"
#include "dcgm_structs.h"
#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMLWLinkTypes.h"
#include "DcgmFMLWLinkError.h"
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
 * The caller fills appropriate member of DcgmLWLinkReq based on the intended
 * LWLink request and the corresponding result is returned in DcgmLWLinkReqResult
 * The requests are asynchronous in nature, means the requests will be returned
 * immediately with a unique requestId (handle) and the caller can check back the
 * status of the request and result using the isLinkReqComplete() method.
 * 
 * Dependency: FMConnInterface - provides methods for sending messages to peer LFM.
 * 
 * Note :  Can't make the members in DcgmLWLinkReq/DcgmLWLinkReqResult as union, 
 * as the individual structures have STL types, which are not POD types.
 */

// node init will be used for data enable, rx cal, set common mode and link init.
typedef struct
{
    uint32 nodeId;
} DcgmLWLinkNodeInitReq;

typedef struct
{
    // nothing specific other than the status, which is part of DcgmLWLinkReqResult
} DcgmLWLinkNodeInitResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkNodeInitStatusReq;

typedef struct
{
    DcgmLinkInitStatusInfoList statusInfo;
} DcgmLWLinkNodeInitStatusResp;

typedef struct
{
    uint32 nodeId;
    uint64 switchId;
    uint64 linkMask;
} DcgmLWLinkNodeInitResetSwitchLinksReq;

typedef struct
{
    // nothing specific other than the status, which is part of DcgmLWLinkReqResult
} DcgmLWLinkNodeInitResetSwitchLinksResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkNodeInitResetAllSwitchLinksReq;

typedef struct
{
    // nothing specific other than the status, which is part of DcgmLWLinkReqResult
} DcgmLWLinkNodeInitResetAllSwitchLinksResp;

typedef struct
{
    uint32 masterNodeId;
    uint64 masterGpuOrSwitchId;
    uint32 masterLinkIndex;
    uint32 slaveNodeId;
    uint64 slaveGpuOrSwitchId;
    uint32 slaveLinkIndex;
    DcgmLWLinkTrainType trainTo;
} DcgmLWLinkConnTrainReq;

typedef struct
{
    DcgmLWLinkStateInfo masterState;
    DcgmLWLinkStateInfo slaveState;
} DcgmLWLinkConnTrainResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkDiscoverIntraNodeConnReq;

typedef struct
{
    // nothing specific other than the status
} DcgmLWLinkDiscoverIntraNodeConnResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkWriteDiscoveryTokenReq;

typedef struct
{
    DcgmLWLinkDiscoveryTokenList tokenInfo;
} DcgmLWLinkWriteDiscoveryTokenResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkReadDiscoveryTokenReq;

typedef struct
{
    DcgmLWLinkDiscoveryTokenList tokenInfo;
} DcgmLWLinkReadDiscoveryTokenResp;

typedef struct
{
    DcgmLWLinkEndPointInfo       localEndInfo;
    DcgmLWLinkRemoteEndPointInfo remoteEndInfo;
} DcgmLWLinkAddInterNodeConnReq;

typedef struct
{
    // nothing specific other than the status
} DcgmLWLinkAddInterNodeConnResp;

typedef struct
{
    uint32 nodeId;
} DcgmLWLinkGetIntraNodeConnReq;

typedef struct
{
    DcgmLWLinkConnList connInfo;
} DcgmLWLinkGetIntraNodeConnResp;

typedef struct 
{
    DcgmLWLinkNodeInitReq                     nodeInitReq;
    DcgmLWLinkNodeInitStatusReq               nodeInitStatusReq;
    DcgmLWLinkNodeInitResetSwitchLinksReq     nodeInitResetSwitchLinksReq;
    DcgmLWLinkNodeInitResetAllSwitchLinksReq  nodeInitResetAllSwitchLinksReq;
    DcgmLWLinkConnTrainReq                    connTrainReq;
    DcgmLWLinkDiscoverIntraNodeConnReq        discoverIntraNodeConnReq;
    DcgmLWLinkReadDiscoveryTokenReq           readDiscoveryTokenReq;
    DcgmLWLinkWriteDiscoveryTokenReq          writeDiscoveryTokenReq;
    DcgmLWLinkAddInterNodeConnReq             addInterNodeConnReq;
    DcgmLWLinkGetIntraNodeConnReq             getIntraNodeConnReq;
} DcgmLWLinkReq;

typedef struct 
{
    uint64                                     requestId;
    DcgmLWLinkErrorCodes                       status;
    DcgmLWLinkNodeInitResp                     nodeInitResp;
    DcgmLWLinkNodeInitStatusResp               nodeInitStatusResp;
    DcgmLWLinkNodeInitResetSwitchLinksResp     nodeInitResetLinkResp;
    DcgmLWLinkNodeInitResetAllSwitchLinksResp  nodeInitResetAllSwitchLinksResp;
    DcgmLWLinkConnTrainResp                    connTrainResp;
    DcgmLWLinkDiscoverIntraNodeConnResp        discoverIntraNodeConnResp;
    DcgmLWLinkReadDiscoveryTokenResp           readDiscoveryTokenResp;
    DcgmLWLinkWriteDiscoveryTokenResp          writeDiscoveryTokenResp;
    DcgmLWLinkAddInterNodeConnResp             addInterNodeConnResp;
    DcgmLWLinkGetIntraNodeConnResp             getIntraNodeConnResp;
} DcgmLWLinkReqResult;

class DcgmFMLWLinkIntf : public FMMessageHandler
{
public:
    DcgmFMLWLinkIntf(FMConnInterface *ctrlConnIntf);

    virtual ~DcgmFMLWLinkIntf();

    dcgmReturn_t sendEnableCommonModeReq(DcgmLWLinkReq &linkReq,
                                         uint64 &requestId);

    dcgmReturn_t sendDisableCommonModeReq(DcgmLWLinkReq &linkReq,
                                          uint64 &requestId);

    dcgmReturn_t sendCalibrateReq(DcgmLWLinkReq &linkReq,
                                  uint64 &requestId);

    dcgmReturn_t sendEnableDataReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    dcgmReturn_t sendInitphase1Req(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);
    
    dcgmReturn_t sendRxInitTermReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);
    
    dcgmReturn_t sendSetRxDetectReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);
    
    dcgmReturn_t sendGetRxDetectReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);
    
    dcgmReturn_t sendInitnegotiateReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId);
#endif

    dcgmReturn_t sendLinkInitReq(DcgmLWLinkReq &linkReq,
                                 uint64 &requestId);

    dcgmReturn_t sendLinkInitStatusReq(DcgmLWLinkReq &linkReq,
                                       uint64 &requestId);

    dcgmReturn_t sendResetSwitchLinksReq(DcgmLWLinkReq &linkReq,
                                         uint64 &requestId);

    dcgmReturn_t sendResetAllSwitchLinksReq(DcgmLWLinkReq &linkReq,
                                            uint64 &requestId);

    dcgmReturn_t sendConnTrainReq(DcgmLWLinkReq &linkReq,
                                  uint64 &requestId);

    dcgmReturn_t sendDiscoverIntraNodeConnReq(DcgmLWLinkReq &linkReq,
                                              uint64 &requestId);

    dcgmReturn_t sendAddInterNodeConnReq(DcgmLWLinkReq &linkReq,
                                         uint64 &requestId);

    dcgmReturn_t sendGetIntraNodeConnReq(DcgmLWLinkReq &linkReq,
                                         uint64 &requestId);

    dcgmReturn_t sendWriteDiscoveryReq(DcgmLWLinkReq &linkReq,
                                       uint64 &requestId);

    dcgmReturn_t sendReadDiscoveryReq(DcgmLWLinkReq &linkReq,
                                      uint64 &requestId);

    bool isLinkReqComplete(uint64 requestId,
                           DcgmLWLinkReqResult &reqResult);

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
        DcgmLWLinkReq req;
        DcgmLWLinkReqResult result;
    } DcgmLWLinkReqCtx;

    dcgmReturn_t sendTrainRequest(lwswitch::FabricManagerMessageType msgType,
                                  DcgmLWLinkReq &linkReq,
                                  uint64 &requestId);

    void genConnTrainReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                            DcgmLWLinkReq &linkReq);

    void genNodeInitReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                           DcgmLWLinkReq &linkReq);

    void genNodeInitStatusReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                 DcgmLWLinkReq &linkReq);

    void genNodeInitResetSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                           DcgmLWLinkReq &linkReq);

    void genNodeInitResetAllSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                              DcgmLWLinkReq &linkReq);

    void genDiscoverIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                        DcgmLWLinkReq &linkReq);

    void genAddInterNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                   DcgmLWLinkReq &linkReq);

    void genGetIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                   DcgmLWLinkReq &linkReq);

    void genWriteDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                 DcgmLWLinkReq &linkReq);

    void genReadDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                DcgmLWLinkReq &linkReq);

    void addToReqPendingTable(uint64 reqId,
                              DcgmLWLinkReq &linkReq,
                              lwswitch::FabricManagerMessageType reqType,
                              uint32 toNodeId);

    void removeFromReqPendingTable(uint64 reqId);

    bool getPendingTrainReq(uint64 reqId, DcgmLWLinkReqCtx &reqCtx);

    void markPendingReqAsComplete(uint64 reqId, DcgmLWLinkReqCtx &reqCtx);

    uint64 getNextTrainReqId(void);

    void handleTrainReqResponseMsg(lwswitch::fmMessage * pFmMessage);

    void handleNodeDisconnect(uint32 nodeId);

    void parseConnTrainResp(lwswitch::lwlinkResponseMsg &rspMsg,
                            DcgmLWLinkReqCtx &reqCtx);

     void parseWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                      DcgmLWLinkReqCtx &reqCtx);

    void parseReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                     DcgmLWLinkReqCtx &reqCtx);

    void parseLinkInitStatusResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                 DcgmLWLinkReqCtx &reqCtx);

    void parseGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg &rspMsg,
                                   DcgmLWLinkReqCtx &reqCtx);

    // debug function helpers

    void dumpTrainCtxEntry(std::ostream *os,
                           uint64 reqId,
                           DcgmLWLinkReqCtx &reqCtx);

    // class members
    typedef std::map <uint64, DcgmLWLinkReqCtx> TrainRequestMap;
    TrainRequestMap  mTrainReqPending;  // outstanding requests
    TrainRequestMap  mTrainReqComplete; // completed requests

    // the maps needs to be protected as it is changed from two thread contexts
    LWOSCriticalSection mLock;

    FMConnInterface *mCtrlConnIntf;
    uint64 mNextReqId;
};
