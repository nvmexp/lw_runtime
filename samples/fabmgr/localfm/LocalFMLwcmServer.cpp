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

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "fm_log.h"

#include "LocalFMLwcmServer.h"

FMServerConn::FMServerConn(FmConnBase* parent, FmServerConnection *pConnection,
                           uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold)
    :FMConnectionBase( parent, rspTimeIntrvl, rspTimeThreshold )
{
    mpParent = parent;
    mpConnection = pConnection;
}

FMServerConn::~FMServerConn()
{

}

struct sockaddr_in
FMServerConn::getRemoteSocketAddr()
{
    struct sockaddr_in remoteAddr = { 0 };
    if ( mpConnection )
    {
        remoteAddr = reinterpret_cast<FmServerConnection*>(mpConnection)->GetRemoteSocketAddr();
    }

    return remoteAddr;
}


LocalFMLwcmServer::LocalFMLwcmServer( FmConnBase* parent, int portNumber, char *sockpath, int isTCP,
                                      uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold )
    : FmSocket( portNumber, sockpath, isTCP, FM_SERVER_WORKER_NUM_THREADS )
{
    mpParent = parent;
    mPortNumber = portNumber;
    mRspTimeIntrvl = rspTimeIntrvl;
    mRspTimeThreshold = rspTimeThreshold;
    
    lwosInitializeCriticalSection(&mMutex);
};

LocalFMLwcmServer::~LocalFMLwcmServer()
{
    closeAllServerConnections();
    // stop our listening thread/server
    StopServer();
    
    lwosDeleteCriticalSection(&mMutex);
};

void
LocalFMLwcmServer::closeAllServerConnections(void)
{
    lwosEnterCriticalSection(&mMutex);
    
    ConnIdToFMServerConnMap::iterator it = mFMServerConnMap.begin();

    while ( it != mFMServerConnMap.end() ) {
        // delete all server connection accepted
        FMServerConn* serverConn = it->second;
        if ( serverConn ) {
            delete serverConn;
            serverConn = NULL;
        }
        it++;
    }
    // clear our mapping information
    mFMServerConnMap.clear();
    
    lwosLeaveCriticalSection(&mMutex);
}

FMIntReturn_t
LocalFMLwcmServer::sendFMMessage(std::string ipAddr, lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    // using the ip address, find the corresponding server connection object
    FMServerConn* serverConn = getFMServerConn( ipAddr );
    if ( serverConn == NULL ) {
        // we don't have a server connection to the specified peer
        FM_LOG_ERROR("invalid server socket object while trying to send message to ip address %s", ipAddr.c_str());
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    return serverConn->sendFMMessage(pFmMessage, trackReq);
}


FMIntReturn_t
LocalFMLwcmServer::sendFMMessage(lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    FMServerConn* serverConn = getFirstConn();
    if ( serverConn == NULL ) {
        FM_LOG_ERROR("invalid server socket object while trying to send message");
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    return serverConn->sendFMMessage(pFmMessage, trackReq);
}

int
LocalFMLwcmServer::OnRequest(fm_request_id_t requestId, FmServerConnection* pConnection)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;

    FmSocketMessage *pMessageRecvd;        /* Pointer to FM Socket Message */
    FmRequest *pFmServerRequest;   /* Pointer to FM Request */
    bool isComplete;                 /* Flag to determine if the request handling is complete */
    int st;                          /* Status code */

    if (!pConnection) {
        FM_LOG_ERROR("socket message receiver handler is called with null socket connection object");
        return -1;
    }

    lwosEnterCriticalSection(&mMutex);
    
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
        FM_LOG_DEBUG("Inactive connection.");
        return 0;
    }

    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr(inet_ntoa(remoteAddr.sin_addr));
    // get the corresponding connection object
    FMServerConn* serverConn = getFMServerConn(strAddr);
    if (serverConn == NULL) {
        // we don't have a server connection to the specified peer
        FM_LOG_ERROR("socket message receiver handler is not able to map the message received to a server socket");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();

        return -1;
    }

    pFmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pFmServerRequest) {
        FM_LOG_DEBUG("Failed to get Info for request id %d.", (int)requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }    

    if (pFmServerRequest->MessageCount() != 1) {
        FM_LOG_DEBUG("Expected single message for the request");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    // Get the message received corresponding to the request id
    pMessageRecvd = pFmServerRequest->GetNextMessage();

    if (NULL == pMessageRecvd) {
        FM_LOG_DEBUG("Failed to get message for request id %d.", (int)requestId);
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
    
    lwosLeaveCriticalSection(&mMutex);
    return 0;
}

void
LocalFMLwcmServer::OnConnectionAdd(fm_connection_id_t connectionId, FmServerConnection *pConnection)
{
    lwosEnterCriticalSection(&mMutex);
    // a new connection is accepted by our server object
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    
    FMServerConn *serverConn = new FMServerConn( mpParent, pConnection, mRspTimeIntrvl, mRspTimeThreshold );

    mFMServerConnMap.insert( std::make_pair(connectionId, serverConn) );

    FM_LOG_DEBUG("Connection add to IP Address %s, port %d, connectionId %d, number of connections %d",
               strAddr.c_str(), mPortNumber, (int)pConnection->GetConnectionId(), (int)mFMServerConnMap.size());
    lwosLeaveCriticalSection(&mMutex);
}

void
LocalFMLwcmServer::OnConnectionRemove(fm_connection_id_t connectionId, FmServerConnection *pConnection)
{
    lwosEnterCriticalSection(&mMutex);
    // an accepted connection is closed. remove from our list as well.
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );

    FMServerConn* serverConn;
    ConnIdToFMServerConnMap::iterator it = mFMServerConnMap.find( connectionId );
    if ( it != mFMServerConnMap.end() ) {
        serverConn = (FMServerConn*)it->second;
        if ( serverConn ) delete serverConn;
    }

    mFMServerConnMap.erase( connectionId );

    FM_LOG_DEBUG("Connection remove IP Address %s, connectionId %d, number of connections %d",
               strAddr.c_str(), (int)pConnection->GetConnectionId(), (int)mFMServerConnMap.size());
    lwosLeaveCriticalSection(&mMutex);
}

FMServerConn *
LocalFMLwcmServer::getFirstConn( )
{
    ConnIdToFMServerConnMap::iterator it;
    if ( mFMServerConnMap.size() == 0 ) {
        // we don't have any server connection
        return NULL;
    }

    it = mFMServerConnMap.begin();

    // get the corresponding connection object
    FMServerConn* serverConn = (FMServerConn*)it->second;
    return serverConn;
}

FMServerConn *
LocalFMLwcmServer::getFMServerConn(std::string ipAddr)
{
    ConnIdToFMServerConnMap::iterator it;
    for ( it = mFMServerConnMap.begin(); it != mFMServerConnMap.end(); it++ ) {
        FMServerConn* serverConn = (FMServerConn*)it->second;
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

