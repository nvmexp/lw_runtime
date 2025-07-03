#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "logging.h"
#include "LwcmRequest.h"
#include "DcgmFMLwcmClient.h"
#include "LwcmClientCallbackQueue.h"

DcgmFMLwcmClientConn::DcgmFMLwcmClientConn(DcgmFMConnBase* parent,
                                           LwcmConnectionHandler* pConnHandler,
                                           LwcmClientCallbackQueue* pClientCQ,
                                           LwcmClientListener* pClientBase,
                                           const char* identifier,
                                           int port_number,
                                           bool addressIsUnixSocket,
                                           int connectionTimeoutMs)
    : LwcmClientConnection(pConnHandler,
                           pClientBase,
                           (char*)identifier,
                           port_number,
                           false,
                           addressIsUnixSocket,
                           connectionTimeoutMs)
    , DcgmFMConnectionBase(parent)
    , mConnectionTimeoutMs(connectionTimeoutMs)
{
    mpParent = parent;
    mConnState = STATE_DISCONNECTED;

    mConnTimer = new DcgmFMTimer( DcgmFMLwcmClientConn::connTimerCB, this );
    mConnTimer->start(1);

    mpConnection = this;
    mpClientCQ = pClientCQ;

    mpClientBase = pClientBase;
}

DcgmFMLwcmClientConn::~DcgmFMLwcmClientConn()
{
    delete mConnTimer;
}

void
DcgmFMLwcmClientConn::connTimerCB(void* ctx)
{
    DcgmFMLwcmClientConn* pObj = (DcgmFMLwcmClientConn*)ctx;
    pObj->onConnTimerExpiry();
}

void
DcgmFMLwcmClientConn::onConnTimerExpiry(void)
{
    retryConnection();
}

void
DcgmFMLwcmClientConn::SignalConnStateChange(ConnectionState state)
{
    PRINT_INFO("%d", "state: %d.", state);
    if ( mpConnection )
    {
        PRINT_INFO("%d, %d", "connectionId %d, connectionState %d.",
                   mpConnection->GetConnectionId(), mpConnection->GetConnectionState());
    }

    /* Dolwmenting the behavior observed for libevent */
    /**
     * If the connection succeeds, LwcmServerConnection::EventCB gets BEV_EVENT_CONNECTED. Implies
     * connection is good to go.
     *
     * When the connection fails due to TCP timeout then LwcmServerConnection::EventCB is notified
     * with BEV_EVENT_ERROR first followed by an immediate additional event of BEV_EVENT_CONNECTED.
     * The additional event BEV_EVENT_CONNECTED doesn't imply that the connection succeeded.
     */

    // This means a special treatment is needed for handling connection/disconnection.
    // DCGM handle this by checking the connection state as a whole instead of bit (see IsConnectionActive())

    switch( state ) {
        case DCGM_CONNECTION_UNKNOWN:
            // we should not end in this state as it is just for initialization
            break;
        case DCGM_CONNECTION_PENDING:
            // We are in the process of connecting
            break;
        case DCGM_CONNECTION_ACTIVE:
            handleConnectEvent();
            break;
        case DCGM_CONNECTION_MARK_TO_CLOSE:
            handleDisconnectEvent();
            break;
        case DCGM_CONNECTION_CLOSED:
            // this state is not used lwrrently
            break;
    }

    /* Call the parent's handler to clean up its state */
    LwcmClientConnection::SignalConnStateChange(state);
}

void
DcgmFMLwcmClientConn::ProcessUnSolicitedMessage(LwcmMessage *pLwcmMessage)
{
    if ( pLwcmMessage == NULL )
        return;

    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pLwcmMessage->GetContent(), pLwcmMessage->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pLwcmMessage ;

    processFMMessage(pFmMessageRcvd);
}

void
DcgmFMLwcmClientConn::ProcessMessage(LwcmMessage *pLwcmMessage)
{
    if ( pLwcmMessage == NULL )
        return;

    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pLwcmMessage->GetContent(), pLwcmMessage->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pLwcmMessage ;

    processFMMessage(pFmMessageRcvd);
}

void
DcgmFMLwcmClientConn::processRequestTimeout()
{
    // we received a request timeout, which means the connection is stuck or invalid
    // clean-up all the pending requests and mark the connection as closed
    // the close event to message handlers will clean-up their pending requests.

    // close the connection from our end
    if (bev) {
        bufferevent_free( bev );
        bev = NULL;
    }

    mConnectionState = DCGM_CONNECTION_MARK_TO_CLOSE;

    // clean requests from LWCM context
    cleanupPendingRequests();

    // self-close won't generate events from libevent.
    // so notify the parent about the disconnection
    mConnState = STATE_DISCONNECTED;
    mpParent->ProcessDisconnect();

    // try to re-establish the connection
    mConnTimer->restart();
}

void
DcgmFMLwcmClientConn::handleDisconnectEvent(void)
{
    ConnState prevState = mConnState;
    mConnState = STATE_DISCONNECTED;

    PRINT_DEBUG("%p %d %d", "handleDisconnectEvent conn %p state %d -> %d",
                this, prevState, mConnState);
    
    if (prevState != mConnState) {
        // delete all the pending requests
        cleanupPendingRequests();
        //stop request tracking timer
        mReqTracker->stopReqTracking();
        // notify parent only once for each connect and disconnect
        mpParent->ProcessDisconnect();
    }
    // try to re-establish the connection
    mConnTimer->restart();
}

struct sockaddr_in
DcgmFMLwcmClientConn::getRemoteSocketAddr()
{
    // return the associated remote (where this client connect to) socket address information
    return m_serv_addr;
}

void
DcgmFMLwcmClientConn::retryConnection(void)
{
    int ret;
    char *ipStr = inet_ntoa( m_serv_addr.sin_addr );

    //reset our internal reading state
    mConnectionState = DCGM_CONNECTION_PENDING;

    // clear the state so that each state bit is marked OFF. See the libevent connection
    // related behavior mentioned in above.
    mReadState = DCGM_CONNECTION_READ_HDR;

    // we need to delete and re-create the buffer event
    // as re-using the buffer event cause no further disconnection events after connect
    if (bev) {
        bufferevent_free( bev );
        bev = NULL;
    }

    // TODO: For modularity, we need to re-factor LwcmClientConnection() constructor
    // so that WaitForConnection() is self sufficient.

    bev = bufferevent_socket_new( mpClientBase->GetBase(), -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE );

    bufferevent_setcb( bev, LwcmClientListener::ReadCB, NULL, LwcmClientConnection::EventCB, this );

    bufferevent_enable( bev, EV_READ|EV_WRITE|EV_TIMEOUT );

    if (mAddressIsUnixSocket) {
        ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_un_addr, sizeof(m_un_addr));
        if (0 != ret) {
            PRINT_ERROR("%s", "Error: Failed to connect to LFM unix domain socket %s ", m_un_addr.sun_path);
            // restart our timer to try the connection again
            mConnTimer->restart();
            return;
        }    
    } else {
         ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_serv_addr, sizeof(m_serv_addr));
        if (0 != ret) {
            PRINT_ERROR("%s %d", "Error: Failed to connect to LFM IP %s Port Number %d", ipStr, ntohs(m_serv_addr.sin_port));
            // restart our timer to try the connection again
            mConnTimer->restart();
            return;
        }    
    }

    if (0 != WaitForConnection(mConnectionTimeoutMs)) {
        if (mAddressIsUnixSocket) {
            PRINT_ERROR("%s", "Error: Timeout while connecting to LFM unix domain socket %s ", m_un_addr.sun_path);
        } else {
            PRINT_ERROR("%s %d", "Error: Timeout while connecting to LFM IP %s Port Number %d", ipStr, ntohs(m_serv_addr.sin_port));        
        }
        // restart our timer to try the connection again
        mConnTimer->restart();
        return;        
    }

    // the connection state should move to active as well.
    if (!IsConnectionActive()) {
        PRINT_ERROR(" ", "Error: Unexpected connection state for LFM ");
    }
}


dcgmReturn_t
DcgmFMLwcmClientConn::sendFMMessageSync(lwswitch::fmMessage *pFmMessage, lwswitch::fmMessage **pResponse)
{
    int ret;
    LwcmMessage lwcmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    unsigned int requestID;
    LwcmRequest *pLwcmClientRequest = NULL;
    lwcm::Command *pCmdWrapper;

    LwcmMessage *pLwcmRecvMsg;
    LwcmProtobuf *pEncodedRxBuf;

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!IsConnectionActive()) {
        DecrReference();
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    // Get Next Request ID
    requestID = GetNextRequestId();

    // Instantiate the LwcmRequest Class
    pLwcmClientRequest = new LwcmRequest(requestID);

    // Add it to map
    AddRequest(requestID, pLwcmClientRequest);

    pFmMessage->set_requestid(requestID);

    /* Copy the protobuf encoded message into a char array */
    msgLength = pFmMessage->ByteSize();
    bufToSend = new char[msgLength];
    pFmMessage->SerializeToArray(bufToSend, msgLength);

    // DEBUG_STDOUT("Length of Message to send to Host Engine:" << msgLength);
    lwcmSendMsg.UpdateMsgHdr(DCGM_MSG_PROTO_REQUEST, requestID, DCGM_PROTO_ST_SUCCESS, msgLength);
    lwcmSendMsg.UpdateMsgContent(bufToSend, msgLength);

    // Send the Message
    ret = SetOutputBuffer(&lwcmSendMsg);

    // free the allocated buffer
    delete[] bufToSend;
    if (DCGM_ST_OK != ret) {
        PRINT_ERROR("%d", "send FM message sync failed with error %d.", ret);
        return DCGM_ST_GENERIC_ERROR;
    }

    // Wait for LWCM Client Request
    ret = pLwcmClientRequest->Wait(60000);
    if (DCGM_ST_OK != ret) {
        return DCGM_ST_GENERIC_ERROR;
    }

    PRINT_DEBUG("%u", "Request Wait completed for request ID: %u", requestID);

    // Get Next Response
    pLwcmRecvMsg = pLwcmClientRequest->GetNextMessage();
    // Initialize Decoder object
    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pLwcmRecvMsg->GetContent(), pLwcmRecvMsg->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pLwcmRecvMsg ;
    pLwcmRecvMsg = NULL;

    *pResponse = pFmMessageRcvd;

    return DCGM_ST_OK;
}

DcgmFMLwcmClient::DcgmFMLwcmClient(DcgmFMConnBase *parent, const char *identifier,
                                   unsigned short port, bool addressIsUnixSocket)
                                          
{
    mpClientListener = new LwcmClientListener();
    mpClientListener->Start();
    mpClientCQ = new LwcmClientCallbackQueue();
    mpConnectionHandler = new LwcmConnectionHandler();
    mpParent = parent;

    mpClientConnection = new DcgmFMLwcmClientConn(mpParent, mpConnectionHandler, mpClientCQ,
                                                  mpClientListener, identifier, port, addressIsUnixSocket );
}

DcgmFMLwcmClient::~DcgmFMLwcmClient()
{
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
    
    /* Stop the client listener from receiving any more packets */
    if (mpClientListener) {
        mpClientListener->StopClientListener();
    }

    if (mpClientCQ) {
        delete mpClientCQ;
        mpClientCQ = NULL;
    }

    if (mpClientListener) {
        int st = mpClientListener->StopAndWait(60000);
        if (st) {
            PRINT_WARNING("", "FMClient Connection: Killing client thread that is still running.");
            mpClientListener->Kill();
        }

        delete mpClientListener;
        mpClientListener = NULL;
    }
}

dcgmReturn_t
DcgmFMLwcmClient::sendFMMessage(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    return mpClientConnection->sendFMMessage(pFmMessage, trackReq);
}

dcgmReturn_t
DcgmFMLwcmClient::sendFMMessageSync(lwswitch::fmMessage *pFmMessage,
                                    lwswitch::fmMessage **pResponse)
{
    return mpClientConnection->sendFMMessageSync(pFmMessage, pResponse);
}
