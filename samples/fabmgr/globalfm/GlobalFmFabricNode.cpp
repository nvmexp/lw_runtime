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
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

#include "GlobalFmFabricNode.h"
#include "GlobalFmHeartbeat.h"
#include "GlobalFabricManager.h"
#include "FMAutoLock.h"
#include "fm_log.h"

/*****************************************************************************/

FMFabricNode::FMFabricNode(const char *identifier, uint32 nodeId,
                               GlobalFabricManager *pGfm,
                               bool addressIsUnixSocket)
{
    lwosInitializeCriticalSection( &mLock );

    mNodeId = nodeId;
    mpGfm = pGfm;
    mConfigError = false;
    mNodeAddress = identifier;

    uint32_t rspTimeIntrvl = pGfm->isSimMode() ? FM_REQ_RESP_TIME_INTRVL_SIM : FM_REQ_RESP_TIME_INTRVL;
    uint32_t rspTimeThreshold = pGfm->isSimMode() ? FM_REQ_RESP_TIME_THRESHOLD_SIM : FM_REQ_RESP_TIME_THRESHOLD;

    // create a tcp client connection with the node's local FM instance
    mpClientConn = new FMLwcmClient( this, identifier, mpGfm->getStartingPort(),
                                     addressIsUnixSocket, rspTimeIntrvl, rspTimeThreshold );

    mpHeartbeat = NULL;

    // don't start heartbeat if the Node's IP is a loopback interface
    // or Unix domain socket is used
    // Note: Not validating a case where the Node's IP is a local network
    // interface's IP address itself instead of loopback. This will require
    // an OS specific API to enumerate all the network interface's IP address.
    std::string strHost(identifier);
    std::string strLoopback(FM_DEFAULT_BIND_INTERFACE);
    if ((addressIsUnixSocket == false) && (strLoopback.compare(strHost) != 0)) {
        // A non-loopback node address. Start the heartbeat tracking class
        mpHeartbeat = new FMGlobalHeartbeat( this, nodeId );
    }
}

FMFabricNode::~FMFabricNode()
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
FMIntReturn_t FMFabricNode::SendControlMessage(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mNodeId );

    return mpClientConn->sendFMMessage( pFmMessage, trackReq );
}

/*****************************************************************************
 * Get a requestId for control message
 *
 *****************************************************************************/
uint32_t FMFabricNode::getControlMessageRequestId(void)
{
    return mpClientConn->mpClientConnection->GetNextRequestId();
}

/*****************************************************************************
 * Send a synchronous FM request message
 * trackReq: true request will be tracked
 *           false request will not be tracked
 *
 *****************************************************************************/
FMIntReturn_t FMFabricNode::SendControlMessageSync(lwswitch::fmMessage *pFmMessage,
                                                   lwswitch::fmMessage **pResponse,
                                                   uint32_t timeoutSec)
{
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mNodeId );
    return mpClientConn->sendFMMessageSync( pFmMessage, pResponse, timeoutSec );
}

int FMFabricNode::ProcessMessage(lwswitch::fmMessage *pFmMessage, bool &isResponse)
{
    // heartbeat is from node context.
    if (pFmMessage->type() == lwswitch::FM_HEARTBEAT_ACK) {
        if (mpHeartbeat) {
            mpHeartbeat->handleHeartbeatAck();
            return FM_INT_ST_OK;
        } else {
            // TODO - heartbeat is not enabled. change this logging when it is enabled.
            FM_LOG_DEBUG("received unexpected heartbeat message from " NODE_ID_LOG_STR " %d", mNodeId);
            return FM_INT_ST_FILE_ILWALID;
        }
    }

    // rest of the message processing is done by individual message handlers.
    return mpGfm->ProcessMessage(mNodeId, pFmMessage, isResponse);
}

void FMFabricNode::ProcessUnSolicitedMessage(lwswitch::fmMessage *pFmMessage)
{
    // heartbeat is from node context.
    if (pFmMessage->type() == lwswitch::FM_HEARTBEAT_ACK) {
        if (mpHeartbeat) {
            mpHeartbeat->handleHeartbeatAck();
            return;
        } else {
            // TODO - heartbeat is not enabled. change this logging when it is enabled.
            FM_LOG_DEBUG("received unexpected heartbeat message from " NODE_ID_LOG_STR " %d", mNodeId);
            return;
        }
    }

    // rest of the message processing is done by individual message handlers.
    bool isResponse;
    mpGfm->ProcessMessage(mNodeId, pFmMessage, isResponse);
}

void FMFabricNode::ProcessConnect(void)
{
    // start heartbeat with this node
    if (mpHeartbeat) {
        mpHeartbeat->startHeartbeat();
    }
    mpGfm->OnFabricNodeConnect(mNodeId);
}

void FMFabricNode::ProcessDisconnect(void)
{
    // stop heartbeat to the node
    if (mpHeartbeat) {
        mpHeartbeat->stopHeartbeat();
    }
    mpGfm->OnFabricNodeDisconnect(mNodeId);
}

bool FMFabricNode::isControlConnectionActive(void)
{
    return mpClientConn->mpClientConnection->IsConnectionActive();
}

void FMFabricNode::setConfigError(void)
{
    FMAutoLock lock(mLock);
    mConfigError = true;
}

void FMFabricNode::clearConfigError(void)
{
    FMAutoLock lock(mLock);
    mConfigError = false;
}

bool FMFabricNode::isConfigError(void)
{
    FMAutoLock lock(mLock);
    return mConfigError;
}

void FMFabricNode::processHeartBeatFailure()
{
    mpGfm->sendDeInitToAllFabricNodes();
}
