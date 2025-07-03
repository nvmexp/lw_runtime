/* 
 * File:   LwcmClientHandler.cpp
 */

#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include "LwcmClientConnection.h"
#include "LwcmClientHandler.h"
#include "lwcm_util.h"
#include "LwcmProtobuf.h"
#include "LwcmProtocol.h"
#include "LwcmSettings.h"
#include "LwcmRequest.h"
#include "LwcmClientCallbackQueue.h"
#include "timelib.h"
#include "logging.h"

/*****************************************************************************
 * Constructor
 *****************************************************************************/
LwcmClientHandler::LwcmClientHandler() {
    mpClientBase = new LwcmClientListener();
    mpClientBase->Start();
    mpClientCQ = new LwcmClientCallbackQueue();
    mpConnectionHandler = new LwcmConnectionHandler();
}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
LwcmClientHandler::~LwcmClientHandler() {

    /* Remove all entries from the connection table */
    if (mpConnectionHandler) {
        delete mpConnectionHandler;
        mpConnectionHandler = NULL;
    }

    /* Stop the client listener from receiving any more packets */
    if (mpClientBase) {
        mpClientBase->StopClientListener();
    }

    if (mpClientCQ) {
        delete mpClientCQ;
        mpClientCQ = NULL;
    }

    /* Shutdown protobuf library at Client side */
    // WAR for bug 2347865
    // google::protobuf::ShutdownProtobufLibrary();

    if (mpClientBase) {
        int st = mpClientBase->StopAndWait(60000);
        if (st) {
            PRINT_WARNING("", "Killing client thread that is still running.");
            mpClientBase->Kill();
        }

        delete mpClientBase;
        mpClientBase = NULL;
    }
}

/*****************************************************************************/
/*
 * helper function for trying to connect to the hostengine.  Returns 0 on success in which
 * case pLwcmHandle will be set as well.
 */
int LwcmClientHandler::tryConnectingToHostEngine(char identifier[],
                                                 unsigned int portNumber,
                                                 dcgmHandle_t* pLwcmHandle,
                                                 bool addressIsUnixSocket,
                                                 int connectionTimeoutMs)
{
    LwcmClientConnection *pLwcmConnection;

    unsigned int connectionId;
    try
    {
        pLwcmConnection = new LwcmClientConnection(GetConnectionHandler(),
                                                   mpClientBase, identifier, portNumber,
                                                   true, addressIsUnixSocket, connectionTimeoutMs);
    } catch (std::runtime_error &e)
    {
        PRINT_ERROR("%s %d", "Got runtime error while connecting to %s:%d", identifier, portNumber);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    if (0 != mpConnectionHandler->AddToConnectionTable(pLwcmConnection, &connectionId)) {
        /* This should never happen as connection ID will be unique */
        PRINT_ERROR("", "AddToConnectionTable returned nonzero");
        delete(pLwcmConnection);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Note: Don't pLwcmConnection->IncrReference() here. This will only be done after the connection
             is established. The connection table is not considered a reference
             in terms of reference counting since it's the backstop for not leaking
             memory rather than an actively-used reference */

    *pLwcmHandle = (dcgmHandle_t)(long long)connectionId;
    return DCGM_ST_OK;
}

/*****************************************************************************
 * Get Connection to the host engine corresponding to the IP address or FQDN
 *****************************************************************************/
dcgmReturn_t LwcmClientHandler::GetConnHandleForHostEngine(char *identifier, 
                                                           dcgmHandle_t *pLwcmHandle,
                                                           unsigned int timeoutMs, 
                                                           bool addressIsUnixSocket)
{
    LwcmConnection *pLwcmConnection = NULL;

    if(!timeoutMs)
        timeoutMs = 5000; /* 5-second default timeout */

    // create local copy of identifier
    char identifierTemp[strlen(identifier + 1)];
    strcpy(identifierTemp,identifier);

    // Parse for port number if specified in identifier
    unsigned int portNumber = 0;
    char * p = NULL;
    
    if(!addressIsUnixSocket)
        p = strchr(identifierTemp,':');

    if (p == NULL)
    {
        portNumber = DCGM_HE_PORT_NUMBER;
    } 
    else 
    {
        *p = '\0'; // breaks up the ip and the port number into two strings

        portNumber = atoi(p + 1);

        // Check if valid
        if((portNumber <= 0) || (portNumber >= 65535)) // 65535 = 2 ^ 16 -1 which is largest possible port number
        {
            return DCGM_ST_BADPARAM;
        }
    }

    unsigned int attempt = 0;
    bool connected = false;
    const unsigned int WAIT_MS = 50;

    timelib64_t start = timelib_usecSince1970();
    timelib64_t now = start;

    for (;;)
    {
        attempt++;
        if (DCGM_ST_OK
            == tryConnectingToHostEngine(identifierTemp, portNumber, pLwcmHandle, addressIsUnixSocket, timeoutMs))
        {
            connected = true;
            break;
        }

        now = timelib_usecSince1970();
        if ((now - start) + WAIT_MS*1000 > timeoutMs*1000)
        {
            break;
        }

        PRINT_DEBUG("%li", "failed connecting to hostengine, still going to try for %li more ms", timeoutMs - (start-now));
        usleep(WAIT_MS * 1000);
    }

    PRINT_DEBUG("%d %li", "finished %d connection attempts to hostengine in about %li ms",
            attempt, (now - start) / 1000);

    if (connected)
    {
        PRINT_DEBUG("", "successfully connected to hostengine");
        return DCGM_ST_OK;
    } else
    {
        PRINT_ERROR("", "failed to connect to hostengine");
        return DCGM_ST_CONNECTION_NOT_VALID;
    }
}

/*****************************************************************************
 * Closes connection to the host engine
 *****************************************************************************/
void LwcmClientHandler::CloseConnForHostEngine(dcgmHandle_t connHandle)
{
    LwcmConnection *pConnection;

    if (NULL == connHandle)
    {
        return;
    }

    pConnection = mpConnectionHandler->GetConnectionEntry((intptr_t)connHandle);
    if (pConnection) {
        pConnection->SetConnectionState(DCGM_CONNECTION_MARK_TO_CLOSE);
        pConnection->RemoveFromConnectionTable();
        pConnection->DecrReference(); /* GetConnectionEntry +1'd this reference */
    }
}

/*****************************************************************************/
LwcmConnectionHandler * LwcmClientHandler::GetConnectionHandler()
{
    return mpConnectionHandler;
}

/*****************************************************************************/
dcgmReturn_t LwcmClientHandler::ExchangeMsgBlocking(dcgmHandle_t connHandle, LwcmProtobuf *pEncodedObj,
        LwcmProtobuf *pDecodeObj, vector<lwcm::Command *> *pRecvdCmds, unsigned int timeout)
{
    int ret;
    LwcmMessage lwcmSendMsg;
    LwcmMessage *pLwcmRecvMsg;
    unsigned int requestID;
    LwcmRequest *pLwcmClientRequest = NULL;
    char *bufToSend;
    unsigned int msgLength;
    LwcmConnection *pLwcmClientConnHandle;

    pLwcmClientConnHandle = mpConnectionHandler->GetConnectionEntry((intptr_t)connHandle);
    if (!pLwcmClientConnHandle) {
        PRINT_ERROR("%p", "could not find host engine connection for handle %p", connHandle);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!pLwcmClientConnHandle->IsConnectionActive()) {
        pLwcmClientConnHandle->DecrReference();
        pLwcmClientConnHandle = 0;
        PRINT_ERROR("%p", "connection for handle %p is no longer active", connHandle);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    /* Get the protobuf encoded message */
    pEncodedObj->GetEncodedMessage(&bufToSend, &msgLength);

    // Get Next Request ID
    requestID = pLwcmClientConnHandle->GetNextRequestId();

    // Instantiate the LwcmClientRequest Class
    pLwcmClientRequest = new LwcmRequest(requestID);

    // Add it to map
    pLwcmClientConnHandle->AddRequest(requestID, pLwcmClientRequest);

    /* Update Encoded Message with a header to be sent over socket */
    lwcmSendMsg.UpdateMsgContent(bufToSend, msgLength);
    lwcmSendMsg.UpdateMsgHdr(DCGM_MSG_PROTO_REQUEST, requestID, DCGM_PROTO_ST_SUCCESS, msgLength);

    // Send the Message
    ret = pLwcmClientConnHandle->SetOutputBuffer(&lwcmSendMsg);
    if (ret < 0) {
        pLwcmClientConnHandle->RemoveRequest(requestID);
        pLwcmClientConnHandle->DecrReference();
        delete pLwcmClientRequest;
        return DCGM_ST_GENERIC_ERROR;
    }

    // Wait for LWCM Client Request
    ret = pLwcmClientRequest->Wait(timeout);
    if (DCGM_ST_OK != ret) {
        PRINT_DEBUG("%d", "pLwcmClientRequest->Wait returned %d", ret);
        pLwcmClientConnHandle->RemoveRequest(requestID);
        pLwcmClientConnHandle->DecrReference();
        delete pLwcmClientRequest;
        return (dcgmReturn_t)ret;
    }

    PRINT_DEBUG("%u", "Request Wait completed for request ID: %u", requestID);

    // Get Next Response
    pLwcmRecvMsg = pLwcmClientRequest->GetNextMessage();

    // Initialize Decoder object
    if (0 != pDecodeObj->ParseRecvdMessage((char *)pLwcmRecvMsg->GetContent(), pLwcmRecvMsg->GetLength(), pRecvdCmds)) {
        PRINT_DEBUG("", "Failed to decode the recvd message for command");
        //PRINT_ERROR("", "Failed to decode the recvd message for command");
        pLwcmClientConnHandle->RemoveRequest(requestID);
        pLwcmClientConnHandle->DecrReference();
        delete pLwcmClientRequest;
        return DCGM_ST_GENERIC_ERROR;
    }

    // Remove it from map
    pLwcmClientConnHandle->RemoveRequest(requestID);

    // delete the request
    delete pLwcmClientRequest;

    // delete the recvd message
    delete pLwcmRecvMsg;

    /* We are no longer looking at pLwcmClientConnHandle */
    pLwcmClientConnHandle->DecrReference();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmClientHandler::ExchangeMsgAsync(dcgmHandle_t dcgmHandle, 
        LwcmProtobuf *pEncodedObj,
        LwcmProtobuf *pDecodeObj, vector<lwcm::Command *> *pRecvdCmds,
        LwcmRequest *request, dcgm_request_id_t *pRequestId)
{
    int ret;
    LwcmMessage lwcmSendMsg;
    LwcmMessage *pLwcmRecvMsg;
    unsigned int requestId;
    char *bufToSend;
    unsigned int msgLength;
    LwcmConnection *pLwcmClientConnHandle;

    if(!request || !dcgmHandle)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    pLwcmClientConnHandle = mpConnectionHandler->GetConnectionEntry((intptr_t)dcgmHandle);
    if (!pLwcmClientConnHandle) 
    {
        PRINT_ERROR("%p", "Bad connection ID %p", dcgmHandle);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    /* Check if the connection is not active at this point. If the connection is
       marked to close then return Connection Not found Error back to the caller */
    if (!pLwcmClientConnHandle->IsConnectionActive()) 
    {
        pLwcmClientConnHandle->DecrReference();
        pLwcmClientConnHandle = 0;
        PRINT_ERROR("%p", "connection ID %p was disconnected", dcgmHandle);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }    

    /* Get the protobuf encoded message */
    pEncodedObj->GetEncodedMessage(&bufToSend, &msgLength);

    // Get Next Request ID
    requestId = pLwcmClientConnHandle->GetNextRequestId();

    /* Set the request ID of our custom handler */
    request->SetRequestId(requestId);

    // Add it to map
    pLwcmClientConnHandle->AddRequest(requestId, request);

    // DEBUG_STDOUT("Length of Message to send to Host Engine:" << msgLength);

    /* Update Encoded Message with a header to be sent over socket */
    lwcmSendMsg.UpdateMsgContent(bufToSend, msgLength);
    lwcmSendMsg.UpdateMsgHdr(DCGM_MSG_PROTO_REQUEST, requestId, 
                             DCGM_PROTO_ST_SUCCESS, msgLength);

    // Send the Message
    ret = pLwcmClientConnHandle->SetOutputBuffer(&lwcmSendMsg);
    if (ret < 0) {
        pLwcmClientConnHandle->DecrReference();
        return DCGM_ST_GENERIC_ERROR;
    }

    // Wait for LWCM Client Request
    ret = request->Wait(60000);
    if (DCGM_ST_OK != ret) {
        pLwcmClientConnHandle->DecrReference();
        return DCGM_ST_GENERIC_ERROR;
    }

    PRINT_DEBUG("%u", "Request Wait completed for request ID: %u", requestId);

    // Get Next Response
    pLwcmRecvMsg = request->GetNextMessage();

    // Initialize Decoder object
    if (0 != pDecodeObj->ParseRecvdMessage((char *)pLwcmRecvMsg->GetContent(), 
                                           pLwcmRecvMsg->GetLength(), pRecvdCmds)) 
    {
        PRINT_ERROR("", "Failed to decode the recvd message for command");
        pLwcmClientConnHandle->DecrReference();
        //PRINT_ERROR("", "Failed to decode the recvd message for command");
        return DCGM_ST_GENERIC_ERROR;
    }    

    /* The entry corresponding to async request is not removed from the map at this 
     * point. The entry will be removed when the request is marked as completed 
     * at later point. The request ID is returned back to the user and should be 
     * passed in back to notify completion and freeing it from the internal storage
     */
    *pRequestId = request->GetRequestId();

    pLwcmClientConnHandle->DecrReference();
    delete pLwcmRecvMsg;
    return DCGM_ST_OK;
}

/*****************************************************************************/


