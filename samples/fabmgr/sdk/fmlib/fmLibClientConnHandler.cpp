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

#include "fm_log.h"
#include "fmlib.pb.h"
#include "timelib.h"
#include "fmLibClientConnHandler.h"
#include "FmClientConnection.h"
#include "FmSocketMessage.h"
#include "FmRequest.h"

using namespace std;

fmLibClientConnHandler::fmLibClientConnHandler()
{
    FM_LOG_DEBUG("Entering fmLibClientConnHandler constructor");

    mClientBase = new FmClientListener();
    mConnectionHandler = new FmConnectionHandler();
    mClientBase->Start();    
}

fmLibClientConnHandler::~fmLibClientConnHandler()
{
    FM_LOG_DEBUG("Entering fmLibClientConnHandler destructor");

    // remove all entries from the connection table
    if (mConnectionHandler) {
        delete mConnectionHandler;
        mConnectionHandler = NULL;
    }

    // stop the client listener from receiving any more packets
    if (mClientBase) {
        mClientBase->StopClientListener();
        int st = mClientBase->StopAndWait(60000);
        if (st) {
            FM_LOG_WARNING("Killing FM client connection thread that is still running");
            mClientBase->Kill();
        }
        delete mClientBase;
        mClientBase = NULL;
    }
}

FMIntReturn_t
fmLibClientConnHandler::openConnToRunningFMInstance(char* addressInfo, 
                                                    fmHandle_t* pConnHandle,
                                                    unsigned int timeoutMs, 
                                                    bool addressIsUnixSocket)
{
    FM_LOG_DEBUG("Entering openConnToRunningFMInstance");

     // create local copy of identifier
    char tempAddress[strlen(addressInfo) + 1];
    strcpy(tempAddress, addressInfo);

    // Parse for port number if specified in identifier
    unsigned int portNumber = 0;
    char* pTemp = NULL;

    if (!addressIsUnixSocket)
        pTemp = strchr(tempAddress,':');

    if (pTemp == NULL) {
        portNumber = FM_CMD_PORT_NUMBER;
    }
    else {
        *pTemp = '\0'; // breaks up the ip and the port number into two strings
        portNumber = atoi(pTemp + 1);
        // check for valid port range.
        // 65535 = 2 ^ 16 -1 which is largest possible port number
        if ((portNumber <= 0) || (portNumber >= 65535)) {
            return FM_INT_ST_BADPARAM;
        }
    }

    unsigned int numAttempt = 0;
    bool bConnected = false;
    const unsigned int WAIT_INTERVAL_MS = 50;
    timelib64_t startTime = timelib_usecSince1970();

    for (;;) {
        numAttempt++;
        FMIntReturn_t retVal = tryConnectingToFMInstance(tempAddress, portNumber,
                                                         pConnHandle, addressIsUnixSocket, timeoutMs);
        if (FM_INT_ST_OK == retVal) {
            bConnected = true;
            break;
        }

        timelib64_t lwrrentTime = timelib_usecSince1970();
        if ((lwrrentTime - startTime) + WAIT_INTERVAL_MS * 1000 > (timeoutMs * 1000)) {
            // timeout over
            FM_LOG_DEBUG("open connection to running FM instance - timeout over \n");
            break;
        }

        // wait for some time before trying again
        FM_LOG_DEBUG("open connection to running FM instance - calling usleep \n");
        usleep(WAIT_INTERVAL_MS * 1000);
    }

    if (bConnected) {
        FM_LOG_DEBUG("open connection to running FM instance - connected\n");
        return FM_INT_ST_OK;
    } else {
        FM_LOG_DEBUG("open connection to running FM instance - failed\n");
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }
}

FMIntReturn_t
fmLibClientConnHandler::closeConnToRunningFMInstance(fmHandle_t connHandle)
{
    if (NULL == connHandle) {
        return FM_INT_ST_BADPARAM;
    }

    FmConnection *pConnection;
    pConnection = mConnectionHandler->GetConnectionEntry((intptr_t)connHandle);
    if (pConnection) {
        pConnection->SetConnectionState(FM_CONNECTION_MARK_TO_CLOSE);
        pConnection->RemoveFromConnectionTable();
        pConnection->DecrReference();
        return FM_INT_ST_OK;
    }

    // no connection for the specified connection handle
    return FM_INT_ST_BADPARAM;
}

FMIntReturn_t
fmLibClientConnHandler::exchangeMsgBlocking(fmHandle_t connHandle, fmlib::Msg *mpFmlibEncodeMsg, 
                                            fmlib::Msg *mpFmlibDecodeMsg, fmlib::Command **pRecvdCmd, 
                                            unsigned int timeout)
{
    int ret;
    FmSocketMessage fmSendMsg;
    FmSocketMessage *pFmRecvMsg;
    unsigned int requestID;
    FmRequest *pFmClientRequest = NULL;
    char *bufToSend;
    unsigned int msgLength;
    FmConnection *pFmClientConnHandle;

    pFmClientConnHandle = mConnectionHandler->GetConnectionEntry((intptr_t)connHandle);
    if (!pFmClientConnHandle) {
        FM_LOG_ERROR("Unable to find socket connection information to FM instance for connection handle %p", connHandle);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    /* Check if the connection is not active at this point. If the connection is
        marked to close then return Connection Not found Error back to the caller */
    if (!pFmClientConnHandle->IsConnectionActive()) {
        pFmClientConnHandle->DecrReference();
        pFmClientConnHandle = 0;
        FM_LOG_ERROR("Unable to find active socket connection information to FM instance for connection handle %p", connHandle);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // /* Get the protobuf encoded message */
    unsigned int length;
    length = mpFmlibEncodeMsg->ByteSize();
    char *mpEncodedMessage = new char [length];
    mpFmlibEncodeMsg->SerializeToArray(mpEncodedMessage, length);
    bufToSend = mpEncodedMessage;
    msgLength = length;
    // pEncodedObj->GetEncodedMessage(&bufToSend, &msgLength);

    // // Get Next Request ID
    requestID = pFmClientConnHandle->GetNextRequestId();

    // // Instantiate the FmClientRequest Class
    pFmClientRequest = new FmRequest(requestID);

    // // Add it to map
    pFmClientConnHandle->AddRequest(requestID, pFmClientRequest);

    // /* Update Encoded Message with a header to be sent over socket */
    fmSendMsg.UpdateMsgContent(bufToSend, msgLength);
    fmSendMsg.UpdateMsgHdr(FM_MSG_PROTO_REQUEST, requestID, FM_PROTO_ST_SUCCESS, msgLength);

    // // Send the Message
    ret = pFmClientConnHandle->SetOutputBuffer(&fmSendMsg);
    if (ret < 0) {
        pFmClientConnHandle->RemoveRequest(requestID);
        pFmClientConnHandle->DecrReference();
        delete pFmClientRequest;
        delete [] mpEncodedMessage;
        return FM_INT_ST_GENERIC_ERROR;
    }

    // // Wait for FM Client Request
    ret = pFmClientRequest->Wait(timeout);
    if (FM_INT_ST_OK != ret) {
        FM_LOG_DEBUG("pFmClientRequest->Wait returned %d", ret);
        pFmClientConnHandle->RemoveRequest(requestID);
        pFmClientConnHandle->DecrReference();
        delete pFmClientRequest;
        delete [] mpEncodedMessage;
        return (FMIntReturn_t)ret;
    }

    FM_LOG_DEBUG("Request Wait completed for request ID: %u", requestID);

    // // Get Next Response
    pFmRecvMsg = pFmClientRequest->GetNextMessage();


    if (true != mpFmlibDecodeMsg->ParseFromArray((char *)pFmRecvMsg->GetContent(), pFmRecvMsg->GetLength())) {
        FM_LOG_DEBUG("Failed to decode the recvd message for command");
        pFmClientConnHandle->RemoveRequest(requestID);
        pFmClientConnHandle->DecrReference();
        delete pFmClientRequest;
        delete pFmRecvMsg;
        delete [] mpEncodedMessage;
        return FM_INT_ST_GENERIC_ERROR;
    }

    *pRecvdCmd = (fmlib::Command*)&mpFmlibDecodeMsg->cmd();
    
    if (*pRecvdCmd == NULL) {
        FM_LOG_ERROR("received response don't have enough command response");
        delete pFmClientRequest;
        delete pFmRecvMsg;
        delete [] mpEncodedMessage;
        return FM_INT_ST_GENERIC_ERROR;
    }

    // Remove it from map
    pFmClientConnHandle->RemoveRequest(requestID);

    // delete the request
    delete pFmClientRequest;

    // delete the recvd message
    delete pFmRecvMsg;

    // delete the send message buffer
    delete [] mpEncodedMessage;

    /* We are no longer looking at pFmClientConnHandle */
    pFmClientConnHandle->DecrReference();

    return FM_INT_ST_OK;
}

FMIntReturn_t
fmLibClientConnHandler::tryConnectingToFMInstance(char* addressIdentifier,
                                                  unsigned int portNumber,
                                                  fmHandle_t* connHandle,
                                                  bool addressIsUnixSocket,
                                                  int connectionTimeoutMs)
{
    FmClientConnection *pFMConnection;
    unsigned int connectionId;

    try {
        pFMConnection = new FmClientConnection(mConnectionHandler, mClientBase,
                                               addressIdentifier, portNumber, true, 
                                               addressIsUnixSocket, connectionTimeoutMs);
    } catch (std::runtime_error &e) {
        FM_LOG_ERROR("Got runtime error while connecting to %s:%d", addressIdentifier, portNumber);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    if (0 != mConnectionHandler->AddToConnectionTable(pFMConnection, &connectionId)) {
        // This should never happen as connection ID will be unique
        FM_LOG_ERROR("failed to add FM connection information to connection table");
        delete(pFMConnection);
        return FM_INT_ST_GENERIC_ERROR;
    }

    *connHandle = (fmHandle_t)(long long)connectionId;
    return FM_INT_ST_OK;
}

