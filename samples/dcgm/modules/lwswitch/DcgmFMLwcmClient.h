#pragma once

#include "LwcmConnection.h"
#include "dcgm_structs.h"
#include "fabricmanager.pb.h"
#include "LwcmClientConnection.h"

#include "DcgmFMCommCtrl.h"
#include "DcgmFMConnectionBase.h"

class LwcmClientCallbackQueue;
class DcgmFMTimer;

class DcgmFMLwcmClientConn : public LwcmClientConnection, public DcgmFMConnectionBase
{

public:
    DcgmFMLwcmClientConn(DcgmFMConnBase* parent, LwcmConnectionHandler *pConnHandler,
                         LwcmClientCallbackQueue *pClientCQ, LwcmClientListener *pClientBase,
                         const char *identifier, int port_number, bool addressIsUnixSocke, int connectionTimeoutMs = 3000);

    virtual ~DcgmFMLwcmClientConn();

    static void connTimerCB(void* ctx);
    void onConnTimerExpiry(void);

    virtual void SignalConnStateChange(ConnectionState state);
    virtual void ProcessUnSolicitedMessage(LwcmMessage *pLwcmMessage);
    virtual void ProcessMessage(LwcmMessage *pLwcmMessage);

    // override the LwcmClientConnection ref counting and self deleting logic
    // as we re-use the connection class for connection/disconnection
    virtual void IncrReference() {  }
    virtual void DecrReference() {  }

    virtual void processRequestTimeout(void);
    virtual struct sockaddr_in getRemoteSocketAddr();

    // this method will send a synchronous message
    // the send will return will a response, or error
    virtual dcgmReturn_t sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                           lwswitch::fmMessage **pResponse);

protected:
    void retryConnection(void);
    void handleDisconnectEvent(void);

    LwcmClientListener      *mpClientBase;
    DcgmFMTimer             *mConnTimer;
    LwcmClientCallbackQueue *mpClientCQ;

    int mConnectionTimeoutMs; //<! Client connection and reconnection timeout in milliseconds
};

/*****************************************************************************
 * We need this parent class instead of combining DcgmFMLwcmClient and 
 * DcgmFMLwcmClientConn as the LwcmClientConnection requires all this 
 * LwcmConnectionHandler, LwcmClientListener etc in constructor itself
 *****************************************************************************/
class DcgmFMLwcmClient
{

public:
    DcgmFMLwcmClient(DcgmFMConnBase *parent, const char *identifier,
                     unsigned short port, bool addressIsUnixSocket);
    virtual ~DcgmFMLwcmClient();

    virtual dcgmReturn_t sendFMMessage(lwswitch::fmMessage *pFmMessage, bool trackReq);
    virtual dcgmReturn_t sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                           lwswitch::fmMessage **pResponse);

    DcgmFMLwcmClientConn    *mpClientConnection;

private:
    LwcmConnectionHandler   *mpConnectionHandler;
    LwcmClientListener      *mpClientListener;
    LwcmClientCallbackQueue *mpClientCQ;
    DcgmFMConnBase          *mpParent;
};

