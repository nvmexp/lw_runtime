#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "DcgmLocalFabricManager.h"
#include "DcgmLocalFabricManagerCoOp.h"
#include "LwcmSettings.h"
#include "LwcmRequest.h"

/*****************************************************************************/
/*                                                                           */
/*  Slave control point for local to local Fabric Manager cooperation        */
/*                                                                           */
/*****************************************************************************/
DcgmFMLocalCoOpServer::DcgmFMLocalCoOpServer(DcgmFMConnBase* pConnBase, char *ip, unsigned short portNumber,
                                             DcgmFMLocalCoOpMgr* parent)
    : DcgmFMLwcmServer(pConnBase, portNumber, ip, true)
{
    mParent = parent;
};

DcgmFMLocalCoOpServer::~DcgmFMLocalCoOpServer()
{
    // do nothing
};

int
DcgmFMLocalCoOpServer::OnRequest(dcgm_request_id_t requestId, LwcmServerConnection* pConnection)
{
    int ret;
    LwcmProtobuf *pEncodedTxBuf;
    LwcmMessage lwcmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    lwcm::Command *pCmdWrapper;

    LwcmMessage *pMessageRecvd;                     /* Pointer to LWCM Message */
    LwcmRequest *pLwcmServerRequest;                /* Pointer to LWCM Request */
    LwcmProtobuf protoObj;                          /* Protobuf object to send or recv the message */
    vector<lwcm::Command *> vecCmds;                /* To store reference to commands inside the protobuf message */
    bool isComplete;                                /* Flag to determine if the request handling is complete */
    int st;                                         /* Status code */

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

    pLwcmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pLwcmServerRequest) {
        DEBUG_STDERR("Failed to get Info for request id " << requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }    

    if (pLwcmServerRequest->MessageCount() != 1) {
        DEBUG_STDERR("Error: Expected single message for the request");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    // Get the message received corresponding to the request id
    pMessageRecvd = pLwcmServerRequest->GetNextMessage();
    if (NULL == pMessageRecvd) {
        DEBUG_STDERR("Failed to get message for request id " << requestId);
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
DcgmFMLocalCoOpClientConn::DcgmFMLocalCoOpClientConn(char* host, unsigned short port, DcgmFMLocalCoOpMgr* parent)
{
    mpFMClient = new DcgmFMLwcmClient(this, host, port, false); // peerLFM is always tcp
    mParent = parent;
};

DcgmFMLocalCoOpClientConn::~DcgmFMLocalCoOpClientConn()
{
    delete mpFMClient;
};

dcgmReturn_t
DcgmFMLocalCoOpClientConn::SendMessage(lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    return mpFMClient->sendFMMessage( pFmMessage, trackReq );
};

int
DcgmFMLocalCoOpClientConn::ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse)
{
    // call Local FM to process this message
    DcgmFMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    return mParent->ProcessPeerNodeRequest( strAddr, pFmMessage, isResponse );
}

void
DcgmFMLocalCoOpClientConn::ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage)
{
    bool isResponse;
    // call Local FM to process this message
    DcgmFMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    std::string strAddr( inet_ntoa(remoteAddr.sin_addr) );
    mParent->ProcessPeerNodeRequest( strAddr, pFmMessage, isResponse);
}

void
DcgmFMLocalCoOpClientConn::ProcessConnect(void)
{
    // established connection with peer LFM. Nothing specific to do
    // as the peer LFM connection is used on demand basis (like for training)
    DcgmFMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    PRINT_INFO("%s", "Info: Connected to peer LFM IP Address %s", strAddr);
}

void
DcgmFMLocalCoOpClientConn::ProcessDisconnect(void)
{
    // disconnected with a peer LFM. Nothing specific to do as the 
    // connection timer internally will try to re-establish the connection.
    DcgmFMLwcmClientConn* clientConn =  mpFMClient->mpClientConnection;
    struct sockaddr_in remoteAddr =  clientConn->getRemoteSocketAddr();
    char* strAddr = inet_ntoa(remoteAddr.sin_addr);
    PRINT_INFO("%s", "Info: Disconnected from peer LFM IP Address %s", strAddr);
}

DcgmFMLocalCoOpMgr::DcgmFMLocalCoOpMgr(DcgmLocalFabricManagerControl* dcgmLFMHndle, char *coOpIp,
                                       unsigned short coOpPortNum)
{

    mSelfNodeId = 0;
    mCoOpPortNum = coOpPortNum;
    mDcgmLFMHndle = dcgmLFMHndle;

    PRINT_DEBUG("%d", "port=%d", coOpPortNum);
    // Create a server object for incoming co-op connections.
    // The actual socket listening (Start()) is based on whether this system
    // has peer LocalFM nodes
    mCoOpServer = new DcgmFMLocalCoOpServer( dcgmLFMHndle, coOpIp, mCoOpPortNum, this );
    mCoOpServerStarted = false;
}

DcgmFMLocalCoOpMgr::~DcgmFMLocalCoOpMgr()
{
    CleanupPeerConnections();
    // stop listening and delete associated server object
    delete mCoOpServer;
}

void
DcgmFMLocalCoOpMgr::setLocalNodeId(uint32 nodeId)
{
    mSelfNodeId = nodeId;
}

void
DcgmFMLocalCoOpMgr::startCoOpServer(void)
{
    if (mCoOpServerStarted) {
        // server object should be in listen mode already
        return;
    }

    mCoOpServer->Start();
    if (0 != mCoOpServer->WaitToStart()) {
        PRINT_ERROR("", "LocalFabricManager: failed to start peerLFM listening socket");
        return;
    }

    mCoOpServerStarted = true;
}

void
DcgmFMLocalCoOpMgr::handleMessage(lwswitch::fmMessage* pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLocalCoOpMgr handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_NODE_INFO_MSG: {
            ProcessNodeInfoMsg( pFmMessage );
            break;
        }
        default: {
            PRINT_WARNING( "", "DcgmFMLocalCoOpMgr received unknown message type\n" );
            break;
        }
    }
}

void
DcgmFMLocalCoOpMgr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    // disconnected with GFM. But nothing specific to do for this FM_NODE_INFO_MSG message handler.
}

int
DcgmFMLocalCoOpMgr::ProcessPeerNodeRequest(std::string nodeIpAddr, lwswitch::fmMessage* pFmMessage, bool &isResponse)
{
    int retVal = 0;
    //TODO - Locking. This is called from client and server socket thread context.
    // See whether we can move the locking to individiual msg handler or to LFM object.
    
    // find the nodeid using the node's ip address information
    NodeIdIpAddrMap::iterator it = mNodeIdIpAddrMap.begin();
    while ( it != mNodeIdIpAddrMap.end() ) {
        if ( it->second == nodeIpAddr) {
            retVal = mDcgmLFMHndle->ProcessPeerLFMMessage(it->first, pFmMessage, isResponse );
            break;
        }
        it++;
    }
    
    return retVal;
}

void
DcgmFMLocalCoOpMgr::ProcessNodeInfoMsg(lwswitch::fmMessage* pFmMessage)
{
    int idx = 0;

    // Node info message is destructive in nature, ie it will delete all the
    // existing peer LFM connections (if any) and re-create them.
    if ( !pFmMessage->has_nodeinfomsg() ) {
        PRINT_WARNING("", "received Node Info message without required fields\n");
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
DcgmFMLocalCoOpMgr::CleanupPeerConnections(void)
{
    // disconnect and erase all the local connections
    NodeIdClientConnMap::iterator cit = mNodeIdClientConnMap.begin();
    while ( cit != mNodeIdClientConnMap.end() ) {
        DcgmFMLocalCoOpClientConn* coOpClient = cit->second;
        mNodeIdClientConnMap.erase( cit++ );
        delete coOpClient;
    }

    // close all the accepted server connections
    mCoOpServer->closeAllServerConnections();
    
    // remove all the nodeid-ip mapping
    mNodeIdIpAddrMap.clear();
}

void
DcgmFMLocalCoOpMgr::CreatePeerConnections(void)
{
    NodeIdIpAddrMap::iterator it = mNodeIdIpAddrMap.begin();
    while ( it != mNodeIdIpAddrMap.end() ) {
        // every peer LFM will initiate CoOp connection to other LFMs with
        // with higher node ids.
        if ( it->first > mSelfNodeId ) {
            std::string strAddr = it->second;
            DcgmFMLocalCoOpClientConn* coOpClient = new DcgmFMLocalCoOpClientConn( (char*)strAddr.c_str(), mCoOpPortNum, this );
            mNodeIdClientConnMap.insert( std::make_pair(it->first, coOpClient) );
        }
        it++;
    }
}

dcgmReturn_t
DcgmFMLocalCoOpMgr::SendMessageToPeerLFM(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool trackReq)
{
    dcgmReturn_t retVal = DCGM_ST_CONNECTION_NOT_VALID;

    // for nodeid higher than local nodeid, we should have client connection
    if ( nodeId > mSelfNodeId ) {
        NodeIdClientConnMap::iterator it = mNodeIdClientConnMap.find( nodeId );
        if (it != mNodeIdClientConnMap.end()) {
            DcgmFMLocalCoOpClientConn* coOpClient = it->second;
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
DcgmFMLocalCoOpMgr::SendNodeInfoMsgAck(lwswitch::fmMessage* pFmReqMessage)
{
    // send the response/ack to GFM
    dcgmReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::nodeInfoAck *ackMsg = new lwswitch::nodeInfoAck();

    ackMsg->set_status( lwswitch::CONFIG_SUCCESS );

    // fill the fabric message
    pFmMessage->set_type( lwswitch::FM_NODE_INFO_ACK );
    pFmMessage->set_allocated_nodeinfoack( ackMsg );
    pFmMessage->set_requestid( pFmReqMessage->requestid() );

    // Send final response to GFM
    retVal = mDcgmLFMHndle->SendMessageToGfm( pFmMessage, false );

    if ( retVal != DCGM_ST_OK ) {
        // can't do much, just log an error
        PRINT_WARNING("", "error while sending train complete message to Fabric Manager\n");
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

