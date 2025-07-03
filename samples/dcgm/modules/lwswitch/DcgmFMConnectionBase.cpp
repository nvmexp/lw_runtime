#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "logging.h"
#include "DcgmFMConnectionBase.h"


DcgmFMConnectionBase::DcgmFMConnectionBase( DcgmFMConnBase *parent )
{
    mConnState = STATE_DISCONNECTED;
    mReqTracker = new DcgmFMReqTracker( this );
    mpConnection = NULL;
    mpParent = parent;
}

DcgmFMConnectionBase::~DcgmFMConnectionBase()
{
    delete mReqTracker;
}

void
DcgmFMConnectionBase::processRequestTimeout(void)
{
    // connection to the peer is timed out.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    PRINT_INFO("%s", "Info: Request timeout to IP Address %s", strAddr);
}

void
DcgmFMConnectionBase::processConnect(void)
{
    // established connection with peer.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    PRINT_INFO("%s", "Info: Connected to IP Address %s", strAddr);
}

void
DcgmFMConnectionBase::processDisconnect(void)
{
    // disconnected with a peer.
    struct sockaddr_in remoteAddr =  getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    PRINT_INFO("%s", "Info: Disconnected from IP Address %s", strAddr);
}

int
DcgmFMConnectionBase::processFMMessage(lwswitch::fmMessage *pFmMessage)
{
    int ret;

    // received an fm message. both server and client side messages received
    // will land here. It is then delivered to the parent, which is LFM for
    // server connection(DcgmLocalFabricManagerControl) and GFM for client
    // side connection (ie FabricNode )

    // update our request tracker
    unsigned int requestId = pFmMessage->requestid();



    // inform our parent about the message completed
    bool isResponse;
    ret = mpParent->ProcessMessage( pFmMessage, isResponse );

    PRINT_DEBUG("%d,%d,%d %d", "DcgmFMConnectionBase: connectionId %d, requestId %d, message Type %d. isResponse=%d",
                mpConnection ? mpConnection->GetConnectionId() : -1,
                requestId, pFmMessage->type(), isResponse);
    if ( isResponse )
    {
        // only remove requestId if the message is a response to a request
        mReqTracker->removeRequest( requestId );
    }

    // complete the request, which will delete the clientRequest
    // if added the client request to dcgm tracker while sending
    // using sendMessage
    if (mpConnection) mpConnection->CompleteRequest(requestId);

    // the pFmMessage is processed by handlers. we can delete the message now
    // if the message handlers requires the message for later processing,
    // they should create a copy of the message
    delete pFmMessage;
    return ret;
}

dcgmReturn_t
DcgmFMConnectionBase::sendFMMessage(lwswitch::fmMessage * pFmMessage, bool trackReq)
{
    int ret;
    LwcmMessage lwcmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    unsigned int requestID;

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!mpConnection || !mpConnection->IsConnectionActive()) {
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    // Get Request ID
    // response message already reuses the request message Request ID
    // new message get the next Request ID from the connection
    requestID = pFmMessage->has_requestid() ? pFmMessage->requestid() : mpConnection->GetNextRequestId();

    PRINT_DEBUG("%d,%d,%d,%d", "DcgmFMConnectionBase: connectionId %d, trackReq %d, requestId %d, message Type %d.",
                mpConnection ? mpConnection->GetConnectionId() : -1,
                trackReq, requestID, pFmMessage->type());
    
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
    lwcmSendMsg.UpdateMsgHdr( DCGM_MSG_PROTO_REQUEST, requestID, DCGM_PROTO_ST_SUCCESS, msgLength );
    lwcmSendMsg.UpdateMsgContent( bufToSend, msgLength );

    // Send the Message
    ret = mpConnection->SetOutputBuffer( &lwcmSendMsg );

    // free the allocated buffer
    delete[] bufToSend;

    if (DCGM_ST_OK != ret) {
        PRINT_ERROR("%d", "send FM message failed with error %d.", ret);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

void
DcgmFMConnectionBase::cleanupPendingRequests(void)
{
    // Note : while sending messages, we add them to DCGM context using
    // LwcmConnection::AddRequest(). When the requests are timed out or
    // when the connection is closed, there is no direct method to delete
    // them. DCGM used to delete the LwcmConnection class itself when the connection
    // is closed. But FM is reusing the connection class for re-connection.

    if (mpConnection) mpConnection->RemoveAllRequests();
}

void
DcgmFMConnectionBase::handleConnectEvent(void)
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
DcgmFMConnectionBase::handleDisconnectEvent(void)
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

