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

#include "FmConnection.h"

#include "FMErrorCodesInternal.h"
#include "fabricmanager.pb.h"
#include "FmClientConnection.h"

#include "FMCommCtrl.h"
#include "FMConnectionBase.h"
#include "FmSocketMessageHdr.h"

#include "lwos.h"

class FMTimer;

class FMLwcmClientConn : public FmClientConnection, public FMConnectionBase
{
public:
    FMLwcmClientConn(FmConnBase* parent, FmConnectionHandler *pConnHandler,
                     FmClientListener *pClientBase, const char *identifier, 
                     int port_number, bool addressIsUnixSocke, uint32_t rspTimeIntrvl,
                     uint32_t rspTimeThreshold, int connectionTimeoutMs = 3000);

    virtual ~FMLwcmClientConn();

    static void connTimerCB(void* ctx);
    void onConnTimerExpiry(void);

    virtual void SignalConnStateChange(ConnectionState state);
    virtual void ProcessUnSolicitedMessage(FmSocketMessage *pFmSocketMsg);
    virtual void ProcessMessage(FmSocketMessage *pFmSocketMsg);

    // override the FmClientConnection ref counting and self deleting logic
    // as we re-use the connection class for connection/disconnection
    virtual void IncrReference() {  }
    virtual void DecrReference() {  }

    virtual void processRequestTimeout(void);
    virtual struct sockaddr_in getRemoteSocketAddr();

    // this method will send a synchronous message
    // the send will return will a response, or error
    virtual FMIntReturn_t sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                            lwswitch::fmMessage **pResponse,
                                            uint32_t timeoutSec);

protected:
    void retryConnection(void);
    void handleDisconnectEvent(void);

    FmClientListener    *mpClientBase;
    FMTimer             *mConnTimer;

    int mConnectionTimeoutMs; ///< Client connection and reconnection timeout in milliseconds

    /*
     * A map keyed by requestId to block and wake up sync message sender
     */
    typedef struct {
        LWOSCriticalSection mutex;
        lwosCV              cond;
        lwswitch::fmMessage *pFmResponse;
    } FMSyncMsgRspCondition_t;

    std::map<fm_request_id_t, FMSyncMsgRspCondition_t*> mRespMsgCondMap;

private:
    FMIntReturn_t waitForSyncResponse(FMSyncMsgRspCondition_t *rspCond,
                                      uint32_t timeoutSec);

    /// Track if a connection ever went active. If so, don't retry when connection is lost
    /// TODO: Might not be needed any more once the retry logic is fully implemented
    bool mPreviouslyActive; 
};

/*****************************************************************************
 * We need this parent class instead of combining FMLwcmClient and 
 * FMLwcmClientConn as the FmClientConnection requires all this 
 * FmConnectionHandler, FmClientListener etc in constructor itself
 *****************************************************************************/
class FMLwcmClient
{

public:
    FMLwcmClient(FmConnBase *parent, const char *identifier,
                 unsigned short port, bool addressIsUnixSocket,
                 uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold);
    virtual ~FMLwcmClient();

    virtual FMIntReturn_t sendFMMessage(lwswitch::fmMessage *pFmMessage, bool trackReq);
    virtual FMIntReturn_t sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                            lwswitch::fmMessage **pResponse,
                                            uint32_t timeoutSec);

    FMLwcmClientConn    *mpClientConnection;

private:
    FmConnectionHandler   *mpConnectionHandler;
    FmClientListener      *mpClientListener;
    FmConnBase            *mpParent;
};

