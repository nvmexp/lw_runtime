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

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif
#include <ctime>

#include "fm_log.h"
#include "FmRequest.h"
#include "FMLwcmClient.h"

FMLwcmClientConn::FMLwcmClientConn(FmConnBase* parent,
                                   FmConnectionHandler* pConnHandler,
                                   FmClientListener* pClientBase,
                                   const char* identifier,
                                   int port_number,
                                   bool addressIsUnixSocket,
                                   uint32_t rspTimeIntrvl,
                                   uint32_t rspTimeThreshold,
                                   int connectionTimeoutMs)
    : FmClientConnection(pConnHandler,
                       pClientBase,
                       (char*)identifier,
                       port_number,
                       false,
                       addressIsUnixSocket,
                       connectionTimeoutMs)
    , FMConnectionBase(parent, rspTimeIntrvl, rspTimeThreshold)
    , mConnectionTimeoutMs(connectionTimeoutMs)
{
    mpParent = parent;
    mConnState = STATE_DISCONNECTED;

    mConnTimer = new FMTimer( FMLwcmClientConn::connTimerCB, this );
    mConnTimer->start(1);

    mpConnection = this;

    mpClientBase = pClientBase;
    mPreviouslyActive = false;
}

FMLwcmClientConn::~FMLwcmClientConn()
{
    delete mConnTimer;
}

void
FMLwcmClientConn::connTimerCB(void* ctx)
{
    FMLwcmClientConn* pObj = (FMLwcmClientConn*)ctx;
    pObj->onConnTimerExpiry();
}

void
FMLwcmClientConn::onConnTimerExpiry(void)
{
    FM_LOG_DEBUG("mConnectionState: %d. retrying connection", mConnectionState);
    retryConnection();
}

void
FMLwcmClientConn::SignalConnStateChange(ConnectionState state)
{
    FM_LOG_DEBUG("SignalConnStateChange state: %d.", state);
    if ( mpConnection )
    {
        FM_LOG_DEBUG("connectionId %d, connectionState %d.",
                      mpConnection->GetConnectionId(), mpConnection->GetConnectionState());
    }

    /* Dolwmenting the behavior observed for libevent */
    /**
     * If the connection succeeds, FmServerConnection::EventCB gets BEV_EVENT_CONNECTED. Implies
     * connection is good to go.
     *
     * When the connection fails due to TCP timeout then FmServerConnection::EventCB is notified
     * with BEV_EVENT_ERROR first followed by an immediate additional event of BEV_EVENT_CONNECTED.
     * The additional event BEV_EVENT_CONNECTED doesn't imply that the connection succeeded.
     */

    // This means a special treatment is needed for handling connection/disconnection.
    // FM handle this by checking the connection state as a whole instead of bit (see IsConnectionActive())

    switch( state ) {
        case FM_CONNECTION_UNKNOWN:
            // we should not end in this state as it is just for initialization
            break;
        case FM_CONNECTION_PENDING:
            // We are in the process of connecting
            break;
        case FM_CONNECTION_ACTIVE:
            handleConnectEvent();
            FM_LOG_DEBUG("FMLwcmClientConn::SignalConnStateChange- stopping retry timer %p", mConnTimer);
            mConnTimer->stop();
            mPreviouslyActive = true;
            break;
        case FM_CONNECTION_MARK_TO_CLOSE:
            handleDisconnectEvent();
            break;
        case FM_CONNECTION_CLOSED:
            // this state is not used lwrrently
            break;
    }

    /* Call the parent's handler to clean up its state */
    FmClientConnection::SignalConnStateChange(state);
}

void
FMLwcmClientConn::ProcessUnSolicitedMessage(FmSocketMessage *pFmSocketMsg)
{
    if ( pFmSocketMsg == NULL )
        return;

    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pFmSocketMsg->GetContent(), pFmSocketMsg->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pFmSocketMsg ;

    // sync message handling sync message receive
    // unblock the receiver who is blocked on this message
    // refer to the sendFMMessageSync()
    if ( pFmMessageRcvd->has_requestid() )
    {
        std::map<fm_request_id_t, FMSyncMsgRspCondition_t*>::iterator it;
        it = mRespMsgCondMap.find(pFmMessageRcvd->requestid());

        if ( it != mRespMsgCondMap.end() )
        {
            FMSyncMsgRspCondition_t *rspCond = it->second;

            // some sync message sender is blocked on this message, unblock it
            lwosEnterCriticalSection(&rspCond->mutex);
            rspCond->pFmResponse = pFmMessageRcvd;
            lwosCondBroadcast(&rspCond->cond);
            lwosLeaveCriticalSection(&rspCond->mutex);

            // sync message sender will process the response
            return;
        }
    }

    // send to message handler to process this response
    processFMMessage(pFmMessageRcvd);
}

void
FMLwcmClientConn::ProcessMessage(FmSocketMessage *pFmSocketMsg)
{
    if ( pFmSocketMsg == NULL )
        return;

    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pFmSocketMsg->GetContent(), pFmSocketMsg->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pFmSocketMsg ;

    processFMMessage(pFmMessageRcvd);
}

void
FMLwcmClientConn::processRequestTimeout()
{
    // we received a request timeout, which means the connection is stuck or invalid
    // clean-up all the pending requests and mark the connection as closed
    // the close event to message handlers will clean-up their pending requests.

    // close the connection from our end
    if (bev) {
        bufferevent_free( bev );
        bev = NULL;
    }

    mConnectionState = FM_CONNECTION_MARK_TO_CLOSE;

    // clean requests from FM context
    cleanupPendingRequests();

    // self-close won't generate events from libevent.
    // so notify the parent about the disconnection
    mConnState = STATE_DISCONNECTED;
    mpParent->ProcessDisconnect();

    // try to re-establish the connection
    mConnTimer->restart();
}

void
FMLwcmClientConn::handleDisconnectEvent(void)
{
    ConnState prevState = mConnState;
    mConnState = STATE_DISCONNECTED;

    FM_LOG_DEBUG("handleDisconnectEvent conn %p state %d -> %d",
                this, prevState, mConnState);
    
    if (prevState != mConnState) {
        // delete all the pending requests
        cleanupPendingRequests();
        //stop request tracking timer
        mReqTracker->stopReqTracking();
        // notify parent only once for each connect and disconnect
        mpParent->ProcessDisconnect();
    }
    // try to re-establish the connection only initially. Once a connection goes down after 
    // going active, don't retry.
    // TODO Remove this condition once the retry logic is correctly and fully implemented
    if (mPreviouslyActive == false )
        mConnTimer->restart();
}

struct sockaddr_in
FMLwcmClientConn::getRemoteSocketAddr()
{
    // return the associated remote (where this client connect to) socket address information
    return m_serv_addr;
}

void
FMLwcmClientConn::retryConnection(void)
{
    int ret;
    char *ipStr = inet_ntoa( m_serv_addr.sin_addr );

    //reset our internal reading state
    mConnectionState = FM_CONNECTION_PENDING;

    // clear the state so that each state bit is marked OFF. See the libevent connection
    // related behavior mentioned in above.
    mReadState = FM_CONNECTION_READ_HDR;

    // we need to delete and re-create the buffer event
    // as re-using the buffer event cause no further disconnection events after connect
    if (bev) {
        bufferevent_free( bev );
        bev = NULL;
    }

    // TODO: For modularity, we need to re-factor FmClientConnection() constructor
    // so that WaitForConnection() is self sufficient.

    bev = bufferevent_socket_new( mpClientBase->GetBase(), -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE );

    bufferevent_setcb( bev, FmClientListener::ReadCB, NULL, FmClientConnection::EventCB, this );

    bufferevent_enable( bev, EV_READ|EV_WRITE|EV_TIMEOUT );

    if (mAddressIsUnixSocket) {
        ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_un_addr, sizeof(m_un_addr));
        if (0 != ret) {
            FM_LOG_ERROR("failed to establish connection to unix domain socket path %s ", m_un_addr.sun_path);
            // restart our timer to try the connection again
            mConnTimer->restart();
            return;
        }    
    } else {
         ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_serv_addr, sizeof(m_serv_addr));
        if (0 != ret) {
            FM_LOG_ERROR("failed to establish connection to ip address %s port number %d", ipStr, ntohs(m_serv_addr.sin_port));
            // restart our timer to try the connection again
            mConnTimer->restart();
            return;
        }    
    }

    if (0 != WaitForConnection(mConnectionTimeoutMs)) {

        if (mAddressIsUnixSocket) {
            // In a single node with UNIX domain socket this is an error because our code starts LFM before starting GFM
            FM_LOG_ERROR("timeout oclwred while waiting for connection to unix domain socket path %s ", 
                         m_un_addr.sun_path);
        } else {
            // In a multi-node system it is possible for GFM to start before LFMs on a different node. 
            // This situation is not an error.
            FM_LOG_DEBUG("timeout oclwred while waiting for connection to ip address %s port number %d",
                         ipStr, ntohs(m_serv_addr.sin_port));        
        }
        // restart our timer to try the connection again
        mConnTimer->restart();
        return;        
    }

    // the connection state should move to active as well.
    if (!IsConnectionActive()) {
        FM_LOG_ERROR("unexpected connection state for socket interface, expected state is ACTIVE ");
    }
}

FMIntReturn_t
FMLwcmClientConn::waitForSyncResponse(FMSyncMsgRspCondition_t *rspCond,
                                      uint32_t timeoutSec)
{
    // Wait for FM Client Response
    FMIntReturn_t rc = FM_INT_ST_OK;

    // block on the condition
    // response message handler would unblock
    lwosEnterCriticalSection(&rspCond->mutex);
    while ( rspCond->pFmResponse == NULL )
    {
        if ( LWOS_TIMEOUT == lwosCondWait(&rspCond->cond, &rspCond->mutex, timeoutSec * 1000) )
        {
            rc = FM_INT_ST_CFG_TIMEOUT;
            break;
        }
    }
    lwosLeaveCriticalSection(&rspCond->mutex);
    return rc;
}

FMIntReturn_t
FMLwcmClientConn::sendFMMessageSync(lwswitch::fmMessage *pFmMessage, lwswitch::fmMessage **pResponse,
                                    uint32_t timeoutSec)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    unsigned int requestID;

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!IsConnectionActive()) {
        DecrReference();
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // Get Next Request ID
    requestID = GetNextRequestId();
    FMSyncMsgRspCondition_t *rspCond = new FMSyncMsgRspCondition_t;

    // Add condition to the map, so that the sender can be woken up when a response is received
    lwosCondCreate(&rspCond->cond);
    lwosInitializeCriticalSection(&rspCond->mutex);
    rspCond->pFmResponse = NULL;
    mRespMsgCondMap.insert(make_pair(requestID, rspCond));

    pFmMessage->set_requestid(requestID);

    /* Copy the protobuf encoded message into a char array */
    msgLength = pFmMessage->ByteSize();
    bufToSend = new char[msgLength];
    pFmMessage->SerializeToArray(bufToSend, msgLength);

    fmSendMsg.UpdateMsgHdr(FM_MSG_PROTO_REQUEST, requestID, FM_PROTO_ST_SUCCESS, msgLength);
    fmSendMsg.UpdateMsgContent(bufToSend, msgLength);

    // Send the Message
    ret = SetOutputBuffer(&fmSendMsg);

    // free the allocated buffer
    delete[] bufToSend;
    if (FM_INT_ST_OK != ret) {
        FM_LOG_ERROR("synchronous fabric manager message send failed with error %d.", ret);
        // remove the condition from the map,
        mRespMsgCondMap.erase(requestID);
        lwosCondDestroy(&rspCond->cond);
        lwosDeleteCriticalSection(&rspCond->mutex);
        delete rspCond;
        return FM_INT_ST_GENERIC_ERROR;
    }

    FMIntReturn_t rc = waitForSyncResponse(rspCond, timeoutSec);
    // get a valid response after being waken up
    if ( ( rc == FM_INT_ST_OK ) && ( rspCond->pFmResponse != NULL ) )
    {
        *pResponse = rspCond->pFmResponse;
    }

    // remove the condition from the map,
    mRespMsgCondMap.erase(requestID);
    lwosCondDestroy(&rspCond->cond);
    lwosDeleteCriticalSection(&rspCond->mutex);
    delete rspCond;

    return rc;
}

FMLwcmClient::FMLwcmClient(FmConnBase *parent, const char *identifier,
                           unsigned short port, bool addressIsUnixSocket,
                           uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold)
                                          
{
    mpClientListener = new FmClientListener();
    mpClientListener->Start();
    mpConnectionHandler = new FmConnectionHandler();
    mpParent = parent;

    mpClientConnection = new FMLwcmClientConn(mpParent, mpConnectionHandler, mpClientListener,
                                              identifier, port, addressIsUnixSocket,
                                              rspTimeIntrvl, rspTimeThreshold );
}

FMLwcmClient::~FMLwcmClient()
{
    /* Stop the client listener from receiving any more packets */
    if (mpClientListener) {
        mpClientListener->StopClientListener();
        
        int st = mpClientListener->StopAndWait(60000);
        if (st) {
            FM_LOG_WARNING("client connection: killing socket listener thread after stop request timeout");
            mpClientListener->Kill();
        }
    }
    
    /* Remove the connection before we remove the client listener to avoid
       the connection having dangling references to the buffered event objects
       that the client listener will automatically free */
    delete mpClientConnection;
    mpClientConnection = 0;

    /* Remove any pending entries with connection table */
    if (mpConnectionHandler) {
        delete mpConnectionHandler;
        mpConnectionHandler = NULL;
    }
    if (mpClientListener) {
        delete mpClientListener;
        mpClientListener = NULL;
    }
}

FMIntReturn_t
FMLwcmClient::sendFMMessage(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    return mpClientConnection->sendFMMessage(pFmMessage, trackReq);
}

FMIntReturn_t
FMLwcmClient::sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                lwswitch::fmMessage **pResponse,
                                uint32_t timeoutSec)
{
    return mpClientConnection->sendFMMessageSync(pFmMessage, pResponse, timeoutSec);
}
