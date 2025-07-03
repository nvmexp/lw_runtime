#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "logging.h"

#include "DcgmFMLwcmServer.h"

DcgmFMLwcmServerConn::DcgmFMLwcmServerConn(DcgmFMConnBase* parent, LwcmServerConnection *pConnection)
    :DcgmFMConnectionBase( parent )
{
    mpParent = parent;
    mpConnection = pConnection;
}

DcgmFMLwcmServerConn::~DcgmFMLwcmServerConn()
{

}

struct sockaddr_in
DcgmFMLwcmServerConn::getRemoteSocketAddr()
{
    struct sockaddr_in remoteAddr = { 0 };
    if ( mpConnection )
    {
        remoteAddr = reinterpret_cast<LwcmServerConnection*>(mpConnection)->GetRemoteSocketAddr();
    }

    return remoteAddr;
}


DcgmFMLwcmServer::DcgmFMLwcmServer( DcgmFMConnBase* parent, int portNumber, char *sockpath, int isTCP )
    : LwcmServer( portNumber, sockpath, isTCP, FM_DCGM_SERVER_WORKER_NUM_THREADS )
{
    mpParent = parent;
    mPortNumber = portNumber;
};

DcgmFMLwcmServer::~DcgmFMLwcmServer()
{
    closeAllServerConnections();
    // stop our listening thread/server
    StopServer();
};

void
DcgmFMLwcmServer::closeAllServerConnections(void)
{
    ConnIdToFMServerConnMap::iterator it = mFMServerConnMap.begin();

    while ( it != mFMServerConnMap.end() ) {
        // delete all server connection accepted
        DcgmFMLwcmServerConn* serverConn = it->second;
        if ( serverConn ) {
            delete serverConn;
            serverConn = NULL;
        }
        it++;
    }
    // clear our mapping information
    mFMServerConnMap.clear();
}

dcgmReturn_t
DcgmFMLwcmServer::sendFMMessage(std::string ipAddr, lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    // using the ip address, find the corresponding server connection object
    DcgmFMLwcmServerConn* serverConn = getFMServerConn( ipAddr );
    if ( serverConn == NULL ) {
        // we don't have a server connection to the specified peer
        PRINT_ERROR(" ", "Invalid connection.");
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    return serverConn->sendFMMessage(pFmMessage, trackReq);
}


dcgmReturn_t
DcgmFMLwcmServer::sendFMMessage(lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    DcgmFMLwcmServerConn* serverConn = getFirstConn();
    if ( serverConn == NULL ) {
        PRINT_ERROR(" ", "Invalid connection.");
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    return serverConn->sendFMMessage(pFmMessage, trackReq);
}

int
DcgmFMLwcmServer::OnRequest(dcgm_request_id_t requestId, LwcmServerConnection* pConnection)
{
    int ret;
    LwcmProtobuf *pEncodedTxBuf;
    LwcmMessage lwcmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    lwcm::Command *pCmdWrapper;

    LwcmMessage *pMessageRecvd;      /* Pointer to LWCM Message */
    LwcmRequest *pLwcmServerRequest; /* Pointer to LWCM Request */
    LwcmProtobuf protoObj;           /* Protobuf object to send or recv the message */
    vector<lwcm::Command *> vecCmds; /* To store reference to commands inside the protobuf message */
    bool isComplete;                 /* Flag to determine if the request handling is complete */
    int st;                          /* Status code */

    if (!pConnection) {
        PRINT_ERROR("", "Invalid connection.");
        return -1;
    }

    /* Add Reference to the connection as the copy is used in this message */
    pConnection->IncrReference();

    /**
     * Before processing the request check if the connection is still in 
     * active state. If the connection is not active then don't even proceed 
     * and mark the request as completed. The CompleteRequest will delete the
     * connection bindings if this request is the last entity holding on to the 
     * connection even when the connection is in inactive state.
     */
    if (!pConnection->IsConnectionActive()) {
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        PRINT_DEBUG("", "Inactive connection.");
        return 0;
    }

    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr(inet_ntoa(remoteAddr.sin_addr));
    // get the corresponding connection object
    DcgmFMLwcmServerConn* serverConn = getFMServerConn(strAddr);
    if (serverConn == NULL) {
        // we don't have a server connection to the specified peer
        PRINT_ERROR("", "Invalid server connection.");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    pLwcmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pLwcmServerRequest) {
        PRINT_DEBUG("%d", "Failed to get Info for request id %d.", (int)requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }    

    if (pLwcmServerRequest->MessageCount() != 1) {
        PRINT_DEBUG(" ", "Expected single message for the request");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    // Get the message received corresponding to the request id
    pMessageRecvd = pLwcmServerRequest->GetNextMessage();
    if (NULL == pMessageRecvd) {
        PRINT_DEBUG("%d", "Failed to get message for request id %d.", (int)requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    } 
    
    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pMessageRecvd->GetContent(), pMessageRecvd->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pMessageRecvd ;
    pMessageRecvd = NULL;

    serverConn->processFMMessage(pFmMessageRcvd);
    pConnection->CompleteRequest(requestId);
    pConnection->DecrReference();
    return 0;
}

void
DcgmFMLwcmServer::OnConnectionAdd(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    // a new connection is accepted by our server object
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    
    DcgmFMLwcmServerConn *serverConn = new DcgmFMLwcmServerConn( mpParent, pConnection );

    mFMServerConnMap.insert( std::make_pair(connectionId, serverConn) );

    PRINT_INFO("%s,%d,%d,%d", "Connection add to IP Address %s, port %d, connectionId %d, number of connections %d",
               strAddr.c_str(), mPortNumber, (int)pConnection->GetConnectionId(), (int)mFMServerConnMap.size());
}

void
DcgmFMLwcmServer::OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    // an accepted connection is closed. remove from our list as well.
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );

    DcgmFMLwcmServerConn* serverConn;
    ConnIdToFMServerConnMap::iterator it = mFMServerConnMap.find( connectionId );
    if ( it != mFMServerConnMap.end() ) {
        serverConn = (DcgmFMLwcmServerConn*)it->second;
        if ( serverConn ) delete serverConn;
    }

    mFMServerConnMap.erase( connectionId );

    PRINT_INFO("%s,%d,%d", "Connection remove IP Address %s, connectionId %d, number of connections %d",
               strAddr.c_str(), (int)pConnection->GetConnectionId(), (int)mFMServerConnMap.size());
}

DcgmFMLwcmServerConn *
DcgmFMLwcmServer::getFirstConn( )
{
    ConnIdToFMServerConnMap::iterator it;
    if ( mFMServerConnMap.size() == 0 ) {
        // we don't have any server connection
        return NULL;
    }

    it = mFMServerConnMap.begin();

    // get the corresponding connection object
    DcgmFMLwcmServerConn* serverConn = (DcgmFMLwcmServerConn*)it->second;
    return serverConn;
}

DcgmFMLwcmServerConn *
DcgmFMLwcmServer::getFMServerConn(std::string ipAddr)
{
    ConnIdToFMServerConnMap::iterator it;
    for ( it = mFMServerConnMap.begin(); it != mFMServerConnMap.end(); it++ ) {
        DcgmFMLwcmServerConn* serverConn = (DcgmFMLwcmServerConn*)it->second;
        struct sockaddr_in remoteAddr =  serverConn->getRemoteSocketAddr();
        std::string strRemoteAddr(inet_ntoa(remoteAddr.sin_addr));
        if (ipAddr.compare(strRemoteAddr) == 0) {
            // found the remote connection
            return serverConn;
        }
    }
    // no connection to the specified ip address
    return NULL;
}

