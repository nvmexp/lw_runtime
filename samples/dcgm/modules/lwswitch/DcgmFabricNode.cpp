
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

#include "DcgmFabricNode.h"
#include "DcgmGlobalHeartbeat.h"
#include "DcgmGlobalControlMsgHndl.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFMAutoLock.h"

/*****************************************************************************/

DcgmFabricNode::DcgmFabricNode(const char *identifier, uint32 nodeId,
                               DcgmGlobalFabricManager *dcgmGFMHndle,
                               bool addressIsUnixSocket)
{
    lwosInitializeCriticalSection( &mLock );

    mNodeId = nodeId;
    mDcgmGFMHndle = dcgmGFMHndle;
    mConfigError = false;

    // create a tcp client connection with the node's local FM instance
    mpClientConn = new DcgmFMLwcmClient( this, identifier, mDcgmGFMHndle->getStartingPort(),
                                         addressIsUnixSocket);

    mpHeartbeat = NULL;

    // don't start heartbeat if the Node's IP is a loopback interface
    // or Unix domain socket is used
    // Note: Not validating a case where the Node's IP is a local network
    // interface's IP address itself instead of loopback. This will require
    // an OS specific API to enumerate all the network interface's IP address.
    std::string strHost(identifier);
    std::string strLoopback(DCGM_HOME_IP);
    if ((addressIsUnixSocket == false) && (strLoopback.compare(strHost) != 0)) {
        // A non-loopback node address. Start the heartbeat tracking class
        mpHeartbeat = new DcgmGlobalHeartbeat( this, nodeId );
    }
}

DcgmFabricNode::~DcgmFabricNode()
{
    if (mpHeartbeat) {
        delete mpHeartbeat;
        mpHeartbeat = NULL;
    }

    if (mpClientConn) {
        delete mpClientConn;
        mpClientConn = NULL;
    }

    lwosDeleteCriticalSection( &mLock );
}

/*****************************************************************************
 * Send an asynchronous FM request message
 * trackReq: true request will be tracked
 *           false request will not be tracked
 *
 *****************************************************************************/
dcgmReturn_t DcgmFabricNode::SendControlMessage(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mNodeId );
    return mpClientConn->sendFMMessage( pFmMessage, trackReq );
}

/*****************************************************************************
 * Get a requestId for control message
 *
 *****************************************************************************/
uint32_t DcgmFabricNode::getControlMessageRequestId(void)
{
    return mpClientConn->mpClientConnection->GetNextRequestId();
}

/*****************************************************************************
 * Send a synchronous FM request message
 * trackReq: true request will be tracked
 *           false request will not be tracked
 *
 *****************************************************************************/
dcgmReturn_t DcgmFabricNode::SendControlMessageSync(lwswitch::fmMessage *pFmMessage,
                                                    lwswitch::fmMessage **pResponse)
{
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mNodeId );
    return mpClientConn->sendFMMessageSync( pFmMessage, pResponse );
}

int DcgmFabricNode::ProcessMessage(lwswitch::fmMessage *pFmMessage, bool &isResponse)
{
    // heartbeat is from node context.
    if (pFmMessage->type() == lwswitch::FM_HEARTBEAT_ACK) {
        if (mpHeartbeat) {
            mpHeartbeat->handleHeartbeatAck();
            return FM_SUCCESS;
        } else {
            PRINT_ERROR("%d", "received unexpected FM heartbeat message from node %d", mNodeId);
            return FM_FILE_ILWALID;
        }
    }

    // rest of the message processing is done by individual message handlers.
    return mDcgmGFMHndle->ProcessMessage(mNodeId, pFmMessage, isResponse);
}

void DcgmFabricNode::ProcessUnSolicitedMessage(lwswitch::fmMessage *pFmMessage)
{
    // heartbeat is from node context.
    if (pFmMessage->type() == lwswitch::FM_HEARTBEAT_ACK) {
        if (mpHeartbeat) {
            mpHeartbeat->handleHeartbeatAck();
            return;
        } else {
            PRINT_ERROR("%d", "received unexpected FM heartbeat message from node %d", mNodeId);
            return;
        }
    }

    // rest of the message processing is done by individual message handlers.
    bool isResponse;
    mDcgmGFMHndle->ProcessMessage(mNodeId, pFmMessage, isResponse);
}

void DcgmFabricNode::ProcessConnect(void)
{
    // start heartbeat with this node
    if (mpHeartbeat) {
        mpHeartbeat->startHeartbeat();
    }
    mDcgmGFMHndle->OnFabricNodeConnect(mNodeId);
}

void DcgmFabricNode::ProcessDisconnect(void)
{
    // stop heartbeat to the node
    if (mpHeartbeat) {
        mpHeartbeat->stopHeartbeat();
    }
    mDcgmGFMHndle->OnFabricNodeDisconnect(mNodeId);
}

bool DcgmFabricNode::isControlConnectionActive(void)
{
    return mpClientConn->mpClientConnection->IsConnectionActive();
}

void DcgmFabricNode::setConfigError(void)
{
    DcgmFMAutoLock lock(mLock);
    mConfigError = true;
}

void DcgmFabricNode::clearConfigError(void)
{
    DcgmFMAutoLock lock(mLock);
    mConfigError = false;
}

bool DcgmFabricNode::isConfigError(void)
{
    DcgmFMAutoLock lock(mLock);
    return mConfigError;
}
