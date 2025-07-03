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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#if defined(_UNIX) && !defined (LW_SUNOS)
#include <arpa/inet.h>
#endif
#include <stdbool.h>
#include <syslog.h>
#include <string.h>

#include "timelib.h"
#include "lwdcommon.h"
#include "prbdec.h"
#include "prblib.h"
#include "fm_internal_api.h"
#include "g_fmInternalLib_pb.h"
#include "fmInternalApiConnHandler.h"

static uint32_t gRequestID = 0;       // to generate FM message request id
static int gConnFd = -1;              // connection socket fd
char mBuf[FM_INTERNAL_MSG_BUF_SIZE];  // socket send and receive buffer

void fmInternalApiConnHandlerInit()
{
    gRequestID = 0;
    gConnFd = -1;
    memset(mBuf, 0, FM_INTERNAL_MSG_BUF_SIZE);
}

FMIntReturn_t exchangeMsgBlocking(PRB_ENCODER *pFmlibEncodeMsg,
                                  PRB_MSG *pFmlibDecodeMsg)
{
#if defined (LW_SUNOS)
    return FM_INT_ST_NOT_SUPPORTED;
#else
    fm_message_header_t msgHdr;
    uint32_t requestID;
    void *pProtoBuf = NULL;

    /* Check if the connection is not active at this point. */
    if (!isConnectedToFMInstance()) {
        syslog(LOG_ERR, "Fabric Manager instance is not connected.\n");
        fprintf(stderr, "Fabric Manager instance is not connected.\n");
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // Get the protobuf encoded message
    uint32_t msgLength = prbEncFinish(pFmlibEncodeMsg, &pProtoBuf);

    // update and copy the message content to the send buffer
    // protobuf message starts after fm_message_header_t
    char *pEncodedMessage = mBuf + sizeof(fm_message_header_t);
    memcpy(pEncodedMessage, pProtoBuf, msgLength);

    // update and copy message header to the send buffer
    msgHdr.msgId = htonl(FM_PROTO_MAGIC);
    msgHdr.requestId = htonl(getNextRequestId());
    msgHdr.msgType = htonl(FM_MSG_PROTO_REQUEST);
    msgHdr.status = htonl(FM_PROTO_ST_SUCCESS);
    msgHdr.length = htonl(msgLength);

    memcpy(mBuf, &msgHdr, sizeof(fm_message_header_t));

    // Send the Message to the socket
    uint32_t lengthToSend = sizeof(fm_message_header_t) + msgLength;
    uint32_t lengthSent = send(gConnFd, mBuf, lengthToSend, 0);
    if (lengthSent != lengthToSend) {
        syslog(LOG_ERR, "failed to send the message to Fabric Manager instance with error %d.\n",
               errno);
        fprintf(stderr, "failed to send the message to Fabric Manager instance with error %d.\n",
                errno);
        return FM_INT_ST_MSG_SEND_ERR;
    }

    // Wait for the response
    int lengthRecved = 0, retryCount;

    for ( retryCount = FM_INTERNAL_MSG_RECV_RETRY; retryCount > 0; retryCount-- ) {
        lengthRecved = recv(gConnFd, mBuf, FM_INTERNAL_MSG_BUF_SIZE, 0);
        if (lengthRecved > (int)sizeof(fm_message_header_t)) {
            // received successfully
            break;
        } else if ((errno == EAGAIN) || (errno = EWOULDBLOCK)) {
            // timed out, retry
            continue;
        } else {
            // other error has oclwrred
            break;
        }
    }

    if (lengthRecved < (int)sizeof(fm_message_header_t)) {
        syslog(LOG_ERR, "received message from Fabric Manager instance is not valid due to header size mismatch.\n");
        fprintf(stderr, "received message from Fabric Manager instance is not valid due to header size mismatch.\n");
        return FM_INT_ST_GENERIC_ERROR;
    }

    // decode the response
    PRB_STATUS prbStatus;
    pProtoBuf = mBuf + sizeof(fm_message_header_t);

    prbStatus = prbDecodeMsg(pFmlibDecodeMsg, (const void *)pProtoBuf, lengthRecved);
    if (PRB_OK != prbStatus) {
        syslog(LOG_ERR,"failed to decode response message from Fabric Manager instance with error %d\n",
               prbStatus);
        fprintf(stderr, "failed to decode response message from Fabric Manager instance with error %d\n",
                prbStatus);
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
#endif
}

FMIntReturn_t connectToFMInstance(unsigned int connTimeoutMs, unsigned int msgTimeoutMs)
{
#if defined (LW_SUNOS)
    return FM_INT_ST_NOT_SUPPORTED;
#else
    int flags = 0, error = 0, ret = 0;
    fd_set rset, wset;
    socklen_t len = sizeof(error);
    struct timeval tv;

    gConnFd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (gConnFd < 0) {
        syslog(LOG_ERR, "request to initialize socket object for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        fprintf(stderr, "request to initialize socket object for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    struct sockaddr_un server;
    server.sun_family = AF_UNIX;
    strcpy(server.sun_path, FM_INTERNAL_API_SOCKET_PATH);

    // set connect timeout
    tv.tv_sec = connTimeoutMs / 1000;
    tv.tv_usec = (connTimeoutMs % 1000) * 1000;

    // clear out descriptor sets for select
    // add socket to the descriptor sets
    FD_ZERO(&rset);
    FD_SET(gConnFd, &rset);
    wset = rset;    //structure assignment ok

    // set socket nonblocking flag
    flags = fcntl(gConnFd, F_GETFL, 0);
    if (flags < 0) {
        syslog(LOG_ERR, "request to get socket for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        fprintf(stderr, "request to get socket for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    if (fcntl(gConnFd, F_SETFL, flags | O_NONBLOCK) < 0) {
        syslog(LOG_ERR, "request to set socket object for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        fprintf(stderr, "request to set socket object for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // initiate non-blocking connect
    ret = connect(gConnFd, (struct sockaddr *) &server, sizeof(struct sockaddr_un));
    if (ret != 0) {
        if (errno != EINPROGRESS) {
            syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            fprintf(stderr, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            return FM_INT_ST_CONNECTION_NOT_VALID;
        }

        // waiting for connect to complete with selecy
        ret = select(gConnFd + 1, &rset, &wset, NULL, (connTimeoutMs) ? & tv : NULL);
        if (ret < 0) {
            syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            fprintf(stderr, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            return FM_INT_ST_CONNECTION_NOT_VALID;
        }

        if (ret == 0) {
            // select timeout
            errno = ETIMEDOUT;
            syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            fprintf(stderr, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            return FM_INT_ST_CONNECTION_NOT_VALID;
        }

        // check for socket error
        if (FD_ISSET(gConnFd, &rset) || FD_ISSET(gConnFd, &wset)) {
            if (getsockopt(gConnFd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
                syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with get socket error %d\n",
                        errno);
                fprintf(stderr, "request to connect with running Fabric Manager instance failed with get socket error %d\n",
                        errno);
                return FM_INT_ST_CONNECTION_NOT_VALID;
            }
        } else {
            syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            fprintf(stderr, "request to connect with running Fabric Manager instance failed with error %d\n",
                    errno);
            return FM_INT_ST_CONNECTION_NOT_VALID;
        }

        if (error) {
            errno = error;
            syslog(LOG_ERR, "request to connect with running Fabric Manager instance failed with socket error %d\n",
                    errno);
            fprintf(stderr, "request to connect with running Fabric Manager instance failed with socket error %d\n",
                    errno);
            return FM_INT_ST_CONNECTION_NOT_VALID;
        }
    }

    // put socket back in blocking mode
    if (fcntl(gConnFd, F_SETFL, flags) < 0) {
        syslog(LOG_ERR, "request to set socket blocking for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        fprintf(stderr, "request to set socket blocking for connecting with running Fabric Manager instance failed with error %d\n",
                errno);
        return FM_INT_ST_CONNECTION_NOT_VALID;
    }

    // set socket message exchange timeout
    tv.tv_sec = msgTimeoutMs / 1000;
    tv.tv_usec = (msgTimeoutMs % 1000) * 1000;
    setsockopt(gConnFd, SOL_SOCKET, SO_RCVTIMEO | SO_SNDTIMEO, (const char*)&tv, sizeof tv);

    return FM_INT_ST_OK;
#endif
}

void disconnectFromFMInstance(void)
{
    if (gConnFd >= 0) {
        close(gConnFd);
        gConnFd = -1;
    }
}

bool isConnectedToFMInstance(void)
{
    if (gConnFd >= 0) {
        return true;
    } else {
        return false;
    }
}

uint32_t getNextRequestId(void)
{
    gRequestID++;
    return gRequestID;
}

void *getConnectionHandle(void)
{
    return (void *)&gConnFd;
}

