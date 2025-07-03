#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "LocalFabricManager.h"
#include "LocalFabricManagerCoOp.h"
#include "FMCommonTypes.h"
#include "FmRequest.h"
#include "fm_log.h"

/*****************************************************************************/
/*                                                                           */
/*  Slave control point for local to local Fabric Manager cooperation        */
/*                                                                           */
/*****************************************************************************/
LocalFMCoOpServer::LocalFMCoOpServer(FmConnBase* pConnBase, char *ip, unsigned short portNumber,
                                     LocalFMCoOpMgr* parent, uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold)
    : LocalFMLwcmServer(pConnBase, portNumber, ip, true, rspTimeIntrvl, rspTimeThreshold)
{
    mParent = parent;
};

LocalFMCoOpServer::~LocalFMCoOpServer()
{
    // do nothing
};

int
LocalFMCoOpServer::OnRequest(fm_request_id_t requestId, FmServerConnection* pConnection)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;

    FmSocketMessage *pMessageRecvd;                /* Pointer to FM Socket Message */
    FmRequest *pFmServerRequest;                     /* Pointer to FM Request */
    vector<fmlib::Command *> vecCmds;                /* To store reference to commands inside the protobuf message */
    bool isComplete;                                 /* Flag to determine if the request handling is complete */
    int st;                                          /* Status code */

    if (!pConnection) {
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
        return 0;
    }

    pFmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pFmServerRequest) {
        FM_LOG_ERROR("peer fm handler: failed to get server socket object for request id %d", requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }    

    if (pFmServerRequest->MessageCount() != 1) {
        FM_LOG_ERROR("peer fm handler: found multiple queued requests for the specified server socket");        
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    // Get the message received corresponding to the request id
    pMessageRecvd = pFmServerRequest->GetNextMessage();
    if (NULL == pMessageRecvd) {
        FM_LOG_ERROR("peer fm handler: failed to get message received on the specified server socket");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    } 
    
    lwswitch::fmMessage *pFmMessageRcvd = new lwswitch::fmMessage();
    pFmMessageRcvd->ParseFromArray( pMessageRecvd->GetContent(), pMessageRecvd->GetLength() );

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pMessageRecvd ;
    pMessageRecvd = NULL;

    // call Local FM to process this message
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    bool isResponse;
    mParent->ProcessPeerNodeRequest(strAddr, pFmMessageRcvd, isResponse);
    delete pFmMessageRcvd;
    pConnection->CompleteRequest(requestId);
    pConnection->DecrReference();

    return 0;
}

/*****************************************************************************/
LocalFMCoOpClientConn::LocalFMCoOpClientConn(char* host, unsigned short port, LocalFMCoOpMgr* parent,
                                             uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold)
{
    mpFMClient = new FMLwcmClient(this, host, port, false, rspTimeIntrvl, rspTimeThreshold); // peerLFM is always tcp
    mParent = parent;
};

LocalFMCoOpClientConn::~LocalFMCoOpClientConn()
{
    delete mpFMClient;
};

FMIntReturn_t
LocalFMCoOpClientConn::SendMessage(lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    return mpFMClient->sendFMMessage( pFmMessage, trackReq );
};

int
LocalFMCoOpClientConn::ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse)
{
    // call Local FM to process this message
    FMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    return mParent->ProcessPeerNodeRequest( strAddr, pFmMessage, isResponse );
}

void
LocalFMCoOpClientConn::ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage)
{
    bool isResponse;
    // call Local FM to process this message
    FMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    mParent->ProcessPeerNodeRequest( strAddr, pFmMessage, isResponse);
}

void
LocalFMCoOpClientConn::ProcessConnect(void)
{
    // established connection with peer LFM. Nothing specific to do
    // as the peer LFM connection is used on demand basis (like for training)
    FMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    FM_LOG_INFO("established socket connection with peer fabric manager ip address %s", strAddr);
}

void
LocalFMCoOpClientConn::ProcessDisconnect(void)
{
    // disconnected with a peer LFM. Nothing specific to do as the 
    // connection timer internally will try to re-establish the connection.
    FMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    FM_LOG_INFO("disconnected socket connection with peer fabric manager ip address %s", strAddr);    
}

LocalFMCoOpMgr::LocalFMCoOpMgr(LocalFabricManagerControl* pLFMHndle, char *coOpIp,
                               unsigned short coOpPortNum)
{

    mSelfNodeId = 0;
    mCoOpPortNum = coOpPortNum;
    mpLFMHndle = pLFMHndle;

    FM_LOG_DEBUG("port=%d", coOpPortNum);
    // Create a server object for incoming co-op connections.
    // The actual socket listening (Start()) is based on whether this system
    // has peer LocalFM nodes

    uint32_t rspTimeIntrvl = mpLFMHndle->isSimMode() ? FM_REQ_RESP_TIME_INTRVL_SIM : FM_REQ_RESP_TIME_INTRVL;
    uint32_t rspTimeThreshold = mpLFMHndle->isSimMode() ? FM_REQ_RESP_TIME_THRESHOLD_SIM : FM_REQ_RESP_TIME_THRESHOLD;

    mCoOpServer = new LocalFMCoOpServer( mpLFMHndle, coOpIp, mCoOpPortNum, this, rspTimeIntrvl, rspTimeThreshold );
    mCoOpServerStarted = false;
}

LocalFMCoOpMgr::~LocalFMCoOpMgr()
{
    CleanupPeerConnections();
    // stop listening and delete associated server object
    delete mCoOpServer;
}

void
LocalFMCoOpMgr::setLocalNodeId(uint32 nodeId)
{
    mSelfNodeId = nodeId;
}

void
LocalFMCoOpMgr::startCoOpServer(void)
{
    if (mCoOpServerStarted) {
        // server object should be in listen mode already
        return;
    }

    mCoOpServer->Start();
    if (0 != mCoOpServer->WaitToStart()) {
        FM_LOG_ERROR("peer fm handler: failed to start peer fabric manager connection socket");
        return;
    }

    mCoOpServerStarted = true;
}

void
LocalFMCoOpMgr::handleMessage(lwswitch::fmMessage* pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMCoOpMgr handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_NODE_INFO_MSG: {
            ProcessNodeInfoMsg( pFmMessage );
            break;
        }
        default: {
        
            FM_LOG_WARNING("peer fm handler: received unknown request typ %d in message handler",
                            pFmMessage->type());
            break;
        }
    }
}

void
LocalFMCoOpMgr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    // disconnected with GFM. But nothing specific to do for this FM_NODE_INFO_MSG message handler.
}

int
LocalFMCoOpMgr::ProcessPeerNodeRequest(std::string nodeIpAddr, lwswitch::fmMessage* pFmMessage, bool &isResponse)
{
    int retVal = 0;
    // TODO - Locking. This is called from client and server socket thread context.
    // See whether we can move the locking to individiual msg handler or to LFM object.
    
    // find the nodeid using the node's ip address information
    NodeIdIpAddrMap::iterator it = mNodeIdIpAddrMap.begin();
    while ( it != mNodeIdIpAddrMap.end() ) {
        if ( it->second == nodeIpAddr) {
            retVal = mpLFMHndle->ProcessPeerLFMMessage(it->first, pFmMessage, isResponse );
            break;
        }
        it++;
    }
    
    return retVal;
}

void
LocalFMCoOpMgr::ProcessNodeInfoMsg(lwswitch::fmMessage* pFmMessage)
{
    int idx = 0;

    // Node info message is destructive in nature, ie it will delete all the
    // existing peer LFM connections (if any) and re-create them.
    if ( !pFmMessage->has_nodeinfomsg() ) {
        FM_LOG_ERROR("peer fm handler: received message don't have all the required fields");
        return;
    }

    // clean existing connections before creating new ones
    CleanupPeerConnections();

    // Copy the add connection values
    lwswitch::nodeInfoMsg infoMsg = pFmMessage->nodeinfomsg();

    // move the server to listening mode only if we have multiple nodes
    if (infoMsg.info_size()) {
        startCoOpServer();
    }

    for ( idx = 0; idx < infoMsg.info_size(); idx++ ) {
        lwswitch::nodeInfo nodeInfo = infoMsg.info(idx);
        mNodeIdIpAddrMap.insert( std::make_pair(nodeInfo.nodeid(), nodeInfo.ipaddress()) );
    }
    
    CreatePeerConnections();
    // send response to GFM
    SendNodeInfoMsgAck (pFmMessage );
}

void
LocalFMCoOpMgr::CleanupPeerConnections(void)
{
    // disconnect and erase all the local connections
    NodeIdClientConnMap::iterator cit = mNodeIdClientConnMap.begin();
    while ( cit != mNodeIdClientConnMap.end() ) {
        LocalFMCoOpClientConn* coOpClient = cit->second;
        mNodeIdClientConnMap.erase( cit++ );
        delete coOpClient;
    }

    // close all the accepted server connections
    mCoOpServer->closeAllServerConnections();
    
    // remove all the nodeid-ip mapping
    mNodeIdIpAddrMap.clear();
}

void
LocalFMCoOpMgr::CreatePeerConnections(void)
{
    NodeIdIpAddrMap::iterator it = mNodeIdIpAddrMap.begin();
    while ( it != mNodeIdIpAddrMap.end() ) {
        // every peer LFM will initiate CoOp connection to other LFMs with
        // with higher node ids.
        if ( it->first > mSelfNodeId ) {
            std::string strAddr = it->second;

            uint32_t rspTimeIntrvl = mpLFMHndle->isSimMode() ? FM_REQ_RESP_TIME_INTRVL_SIM : FM_REQ_RESP_TIME_INTRVL;
            uint32_t rspTimeThreshold = mpLFMHndle->isSimMode() ? FM_REQ_RESP_TIME_THRESHOLD_SIM : FM_REQ_RESP_TIME_THRESHOLD;

            LocalFMCoOpClientConn* coOpClient = new LocalFMCoOpClientConn( (char*)strAddr.c_str(), mCoOpPortNum, this,
                                                                            rspTimeIntrvl, rspTimeThreshold );
            mNodeIdClientConnMap.insert( std::make_pair(it->first, coOpClient) );
        }
        it++;
    }
}

FMIntReturn_t
LocalFMCoOpMgr::SendMessageToPeerLFM(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    FMIntReturn_t retVal = FM_INT_ST_CONNECTION_NOT_VALID;

    // for nodeid higher than local nodeid, we should have client connection
    if ( nodeId > mSelfNodeId ) {
        NodeIdClientConnMap::iterator it = mNodeIdClientConnMap.find( nodeId );
        if (it != mNodeIdClientConnMap.end()) {
            LocalFMCoOpClientConn* coOpClient = it->second;
            retVal = coOpClient->SendMessage( pFmMessage, trackReq );
        }
    } else {
        // for nodeid less than local nodeid, we should have server connection
        // find the node's ip address and request server object to send the message
        NodeIdIpAddrMap::iterator it = mNodeIdIpAddrMap.find( nodeId );
        if (it != mNodeIdIpAddrMap.end()) {
            retVal = mCoOpServer->sendFMMessage(it->second, pFmMessage, true);
        }
    }
    return retVal;
}

void
LocalFMCoOpMgr::SendNodeInfoMsgAck(lwswitch::fmMessage* pFmReqMessage)
{
    // send the response/ack to GFM
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::nodeInfoAck *ackMsg = new lwswitch::nodeInfoAck();

    ackMsg->set_status( lwswitch::CONFIG_SUCCESS );

    // fill the fabric message
    pFmMessage->set_type( lwswitch::FM_NODE_INFO_ACK );
    pFmMessage->set_allocated_nodeinfoack( ackMsg );
    pFmMessage->set_requestid( pFmReqMessage->requestid() );

    // Send final response to GFM
    retVal = mpLFMHndle->SendMessageToGfm( pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // can't do much, just log an error
        FM_LOG_ERROR("peer fm handler: error while sending request complete response message to fabric manager");
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMCoOpMgr::getPeerNodeIds(std::set<uint32> &nodeIds)
{
    nodeIds.clear();
    for ( auto it = mNodeIdIpAddrMap.begin(); it != mNodeIdIpAddrMap.end(); it++ )
    {
        nodeIds.insert ( it->first );
    }
    return;
}
#endif
