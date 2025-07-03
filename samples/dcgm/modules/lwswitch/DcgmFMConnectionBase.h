#pragma once

#include "LwcmConnection.h"

#include "dcgm_structs.h"

#include "LwcmProtobuf.h"
#include "fabricmanager.pb.h"

#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMTimer.h"
#include "DcgmFMReqTracker.h"

// Base class abstraction client and server connection
// that implements asynchronous communication

class DcgmFMReqTracker;

class DcgmFMConnectionBase
{
public:
    DcgmFMConnectionBase( DcgmFMConnBase *parent );
    virtual ~DcgmFMConnectionBase();

    virtual void processRequestTimeout(void);
    virtual void processConnect(void);
    virtual void processDisconnect(void);

    // this method will be called when we receive a message on client connection without any active request
    // since DCGM is req-resp semantics, this is an attempt to de-couple that req-resp semantics.
    // eventually we should have only one of this process message route.
    virtual int processFMMessage(lwswitch::fmMessage *pFmMessage);

    // this method will send an asynchronous message
    // trackReq: true if a response is expected,
    //           false if no response is expected
    virtual dcgmReturn_t sendFMMessage(lwswitch::fmMessage * pFmMessage, bool trackReq);

    virtual void cleanupPendingRequests(void);

    virtual struct sockaddr_in getRemoteSocketAddr() = 0;

    LwcmConnection   *mpConnection;

protected:
    typedef enum
    {
        STATE_CONNECTED = 1,
        STATE_DISCONNECTED = 2,
    } ConnState;

    virtual void handleConnectEvent(void);
    virtual void handleDisconnectEvent(void);

    ConnState         mConnState;
    DcgmFMReqTracker *mReqTracker;
    DcgmFMConnBase   *mpParent;
};


