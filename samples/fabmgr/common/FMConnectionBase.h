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

#include "FmConnection.h"

#include "FMErrorCodesInternal.h"

#include "fabricmanager.pb.h"

#include "FMCommonTypes.h"
#include "FMCommCtrl.h"
#include "FMTimer.h"
#include "FMReqTracker.h"

// Base class abstraction client and server connection
// that implements asynchronous communication

class FMReqTracker;

class FMConnectionBase
{
public:
    FMConnectionBase( FmConnBase *parent, uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold );
    virtual ~FMConnectionBase();

    virtual void processRequestTimeout(void);
    virtual void processConnect(void);
    virtual void processDisconnect(void);

    // this method will be called when we receive a message on client connection without any active request
    // since FM is req-resp semantics, this is an attempt to de-couple that req-resp semantics.
    // eventually we should have only one of this process message route.
    virtual int processFMMessage(lwswitch::fmMessage *pFmMessage);

    // this method will send an asynchronous message
    // trackReq: true if a response is expected,
    //           false if no response is expected
    virtual FMIntReturn_t sendFMMessage(lwswitch::fmMessage * pFmMessage, bool trackReq);

    virtual void cleanupPendingRequests(void);

    virtual struct sockaddr_in getRemoteSocketAddr() = 0;

    FmConnection   *mpConnection;

protected:
    typedef enum
    {
        STATE_CONNECTED = 1,
        STATE_DISCONNECTED = 2,
    } ConnState;

    virtual void handleConnectEvent(void);
    virtual void handleDisconnectEvent(void);

    ConnState     mConnState;
    FMReqTracker *mReqTracker;
    FmConnBase   *mpParent;
};


