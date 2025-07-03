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
#ifdef __linux__ 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif
#include "fm_log.h"
#include "FMConnectionBase.h"


FMConnectionBase::FMConnectionBase( FmConnBase *parent, uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold )
{
    mConnState = STATE_DISCONNECTED;

    mReqTracker = new FMReqTracker( this, rspTimeIntrvl, rspTimeThreshold );
    mpConnection = NULL;
    mpParent = parent;
}

FMConnectionBase::~FMConnectionBase()
{
    delete mReqTracker;
}

void
FMConnectionBase::processRequestTimeout(void)
{
    // connection to the peer is timed out.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    FM_LOG_INFO("connection base: request over socket interface address %s timed out", strAddr);
}

void
FMConnectionBase::processConnect(void)
{
    // established connection with peer.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    FM_LOG_INFO("connection base: connected to socket interface address %s", strAddr);
}

void
FMConnectionBase::processDisconnect(void)
{
    // disconnected with a peer.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    FM_LOG_INFO("connection base: disconnected from socket inteface address %s", strAddr);
}

int
FMConnectionBase::processFMMessage(lwswitch::fmMessage *pFmMessage)
{
    int ret;

    // received an fm message. both server and client side messages received
    // will land here. It is then delivered to the parent, which is LFM for
    // server connection(LocalFabricManagerControl) and GFM for client
    // side connection (ie FabricNode )

    // update our request tracker
    unsigned int requestId = pFmMessage->requestid();
    // inform our parent about the message completed
    bool isResponse;
    ret = mpParent->ProcessMessage( pFmMessage, isResponse );

    //FM_LOG_DEBUG("FMConnectionBase: connectionId %d, requestId %d, message Type %d. isResponse=%d",
                //mpConnection ? mpConnection->GetConnectionId() : -1,
                //requestId, pFmMessage->type(), isResponse);
    if ( isResponse )
    {
        // only remove requestId if the message is a response to a request
        mReqTracker->removeRequest( requestId );
    }

    // complete the request, which will delete the clientRequest
    // if added the client request to fm tracker while sending
    // using sendMessage
    if (mpConnection) mpConnection->CompleteRequest(requestId);

    // the pFmMessage is processed by handlers. we can delete the message now
    // if the message handlers requires the message for later processing,
    // they should create a copy of the message
    delete pFmMessage;
    return ret;
}

FMIntReturn_t
FMConnectionBase::sendFMMessage(lwswitch::fmMessage * pFmMessage, bool trackReq)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    unsigned int requestID;

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!mpConnection || !mpConnection->IsConnectionActive()) {
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // Get Request ID
    // response message already reuses the request message Request ID
    // new message get the next Request ID from the connection
    requestID = pFmMessage->has_requestid() ? pFmMessage->requestid() : mpConnection->GetNextRequestId();

    //FM_LOG_DEBUG("FMConnectionBase: connectionId %d, trackReq %d, requestId %d, message Type %d.",
                //mpConnection ? mpConnection->GetConnectionId() : -1,
                //trackReq, requestID, pFmMessage->type());
    
    if (trackReq) {
        // Add requestID to tracker
        mReqTracker->addRequest( requestID );
    }

    pFmMessage->set_requestid( requestID );

    /* Copy the protobuf encoded message into a char array */
    msgLength = pFmMessage->ByteSize();

    bufToSend = new char[msgLength];
    pFmMessage->SerializeToArray( bufToSend, msgLength );

    // DEBUG_STDOUT("Length of Message to send to Host Engine:" << msgLength);
    fmSendMsg.UpdateMsgHdr( FM_MSG_PROTO_REQUEST, requestID, FM_PROTO_ST_SUCCESS, msgLength );
    fmSendMsg.UpdateMsgContent( bufToSend, msgLength );

    // Send the Message
    ret = mpConnection->SetOutputBuffer( &fmSendMsg );

    // free the allocated buffer
    delete[] bufToSend;

    if (FM_INT_ST_OK != ret) {
        FM_LOG_ERROR("connection base: send request over socket inteface failed with error %d.", ret);
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

void
FMConnectionBase::cleanupPendingRequests(void)
{
    // Note : while sending messages, we add them to FM context using
    // FmConnection::AddRequest(). When the requests are timed out or
    // when the connection is closed, there is no direct method to delete
    // them. FM used to delete the FmConnection class itself when the connection
    // is closed. But FM is reusing the connection class for re-connection.

    if (mpConnection) mpConnection->RemoveAllRequests();
}

void
FMConnectionBase::handleConnectEvent(void)
{
    ConnState prevState = mConnState;
    mConnState = STATE_CONNECTED;
    if (prevState != mConnState) {
        // notify parent only once for each connect and disconnect
        mpParent->ProcessConnect();
        //start request tracking timer
        mReqTracker->startReqTracking();
    }
}

void
FMConnectionBase::handleDisconnectEvent(void)
{
    ConnState prevState = mConnState;
    mConnState = STATE_DISCONNECTED;
    if (prevState != mConnState) {
        // delete all the pending requests
        cleanupPendingRequests();
        //stop request tracking timer
        mReqTracker->stopReqTracking();
        // notify parent only once for each connect and disconnect
        mpParent->ProcessDisconnect();
    }
}

