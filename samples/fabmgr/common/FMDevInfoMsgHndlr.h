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
#include <iostream>
#include <fstream>
#include "FMErrorCodesInternal.h"
#include "fabricmanager.pb.h"
#include "FMCommCtrl.h"
#include "FMLWLinkDeviceRepo.h"

/*****************************************************************************/
/*  Fabric Manager various device information query/resp message handler     */
/*****************************************************************************/

/*
 * FMDevInfoMsgHdlrBase:
 *  This class builds the framework for handling the Device information request.
 *  The GPB message is send from GFM and each LFM will respond to the
 *  request with locally device's detailed information.
 *
 * LocalFMDevInfoMsgHdlr
 *  Represents the message handler in LFM context and handles the request from GFM.
 *  The response is prepared as another GPB message and send back to GFM. Handles
 *  the following requests,
 *  FM_NODE_GET_SWITCH_DEVICE_INFO_REQ - To query all the detected lwswitch devices
 *  FM_NODE_GET_GPU_DEVICE_INFO_REQ - To query all the detected GPU devices
 *  FM_NODE_GET_LWLINK_DEVICE_INFO_REQ - To query all the LWLink device details.
 * 
 *  Note: Ideally LWLINK_DEVICE is a super set of all the detected LWSwitches and GPUs.
 *  But, making all of them as individual requests as the information is retrieved from
 *  three different modules (ie drivers, RM for GPU, LWSwtichDrv for Switch and
 *  LWLinkDrv for lwlink). This will work as it is even if we make separate driver for
 *  for each of these devices as opposed to single lwpu.ko today.
 *
 * GlobalFMDevInfoMsgHdlr
 *  Represents the message handler in the GFM context and provides an interface for
 *  sending various device information requests asynchronously.
 */

class LocalFabricManagerControl;

class FMDevInfoMsgHdlrBase : public FMMessageHandler
{
public:

    FMDevInfoMsgHdlrBase(FMConnInterface *ctrlConnIntf);

    virtual ~FMDevInfoMsgHdlrBase();

    // FMMessageHandler overrides for handling FM messages from GFM
    virtual void handleMessage(lwswitch::fmMessage *pFmMessage);

private:
    // virtual methods for each request and response type
    // which will be implemented by LFM or GFM accordingly.
    virtual void onLWLinkDevInfoReq(lwswitch::fmMessage *pFmMessage) { };
    virtual void onLWLinkDevInfoRsp(lwswitch::fmMessage *pFmMessage) { };
    virtual void onLWSwitchDevInfoReq(lwswitch::fmMessage *pFmMessage) { };
    virtual void onLWSwitchDevInfoRsp(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGpuDevInfoReq(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGpuDevInfoRsp(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGpuLWLinkSpeedInfoReq(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGpuLWLinkSpeedInfoRsp(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGetFMVersionInfoReq(lwswitch::fmMessage *pFmMessage) { };
    virtual void onGetFMVersionInfoRsp(lwswitch::fmMessage *pFmMessage) { };

protected:
    FMConnInterface *mCtrlConnIntf;
};

class LocalFMDevInfoMsgHdlr : public FMDevInfoMsgHdlrBase
{
public:

    LocalFMDevInfoMsgHdlr(LocalFabricManagerControl *pLfm);

    virtual ~LocalFMDevInfoMsgHdlr();

    // FMMessageHandler overrides for handling FM messages from GFM
    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

private:
    virtual void onLWLinkDevInfoReq(lwswitch::fmMessage *pFmMessage);
    virtual void onLWSwitchDevInfoReq(lwswitch::fmMessage *pFmMessage);
    virtual void onGpuDevInfoReq(lwswitch::fmMessage *pFmMessage);
    virtual void onGpuLWLinkSpeedInfoReq(lwswitch::fmMessage *pFmMessage);
    virtual void onGetFMVersionInfoReq(lwswitch::fmMessage *pFmMessage);

    LocalFabricManagerControl *mpLfm;
};

typedef struct 
{
    uint64                          infoReqId;
    uint32                          status;
    lwswitch::fmMessage             devInfoRspMsg;
}DevInfoReqResult;

class GlobalFMDevInfoMsgHdlr : public FMDevInfoMsgHdlrBase
{
public:

    GlobalFMDevInfoMsgHdlr(FMConnInterface *ctrlConnIntf);

    virtual ~GlobalFMDevInfoMsgHdlr();

    // FMMessageHandler overrides for handling FM messages from GFM
    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    FMIntReturn_t sendLWLinkDevInfoReq(uint32 nodeId, uint64 &requestId);
    FMIntReturn_t sendLWSwitchDevInfoReq(uint32 nodeId, uint64 &requestId);
    FMIntReturn_t sendGpuDevInfoReq(uint32 nodeId, uint64 &requestId);
    FMIntReturn_t sendGpuLWLinkSpeedInfoReq(uint32 nodeId, uint64 &requestId);
    FMIntReturn_t sendFMVersionInfoReq(uint32 nodeId, uint64 &requestId);

    bool isDevInfoReqComplete(uint64 &requestId, DevInfoReqResult &reqResult);

private:

    typedef struct
    {
        uint32 toNodeId;
        DevInfoReqResult result;
    } FmDevInfoReqCtx;

    virtual void onLWLinkDevInfoRsp(lwswitch::fmMessage *pFmMessage);
    virtual void onLWSwitchDevInfoRsp(lwswitch::fmMessage *pFmMessage);
    virtual void onGpuDevInfoRsp(lwswitch::fmMessage *pFmMessage);
    virtual void onGpuLWLinkSpeedInfoRsp(lwswitch::fmMessage *pFmMessage);
    virtual void onGetFMVersionInfoRsp(lwswitch::fmMessage *pFmMessage);

    FMIntReturn_t sendDevInfoRequest(lwswitch::FabricManagerMessageType msgType,
                                    uint32 nodeId, uint64 &requestId);

    void handleDevInfoReqCompletion(lwswitch::fmMessage *pFmMessage);
    void addToReqPendingTable(uint64 reqId, uint32 toNodeId);
    void removeFromReqPendingTable(uint64 reqId);
    bool getPendingInfoReq(uint64 reqId, FmDevInfoReqCtx &reqCtx);
    void markPendingReqAsComplete(uint64 reqId, FmDevInfoReqCtx &reqCtx);

    void handleNodeDisconnect(uint32 nodeId);
    uint64 getNextDevInfoReqId(void);

    typedef std::map <uint64, FmDevInfoReqCtx> DevInfoRequestMap;
    DevInfoRequestMap mDevInfoReqPending;  // outstanding requests
    DevInfoRequestMap mDevInfoReqComplete; // completed requests
    uint64 mNextInfoReqId;
    // the maps needs to be protected as it is changed from two thread contexts
    LWOSCriticalSection mLock;
};
