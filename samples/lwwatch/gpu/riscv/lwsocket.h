/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#ifndef _LWSOCKET_H_
#define _LWSOCKET_H_

#include <stdlib.h>
#include "lwstatus.h"

/**
  Platform independent (TCP) connection interface
 */
typedef struct LwSocketPair LwSocketPair;

/**
 * @brief lwSocketCreate Creates LwSocketPair. May internally initialize socket subsystem.
 * @return Null on error.
 */
LwSocketPair *lwSocketCreate(void);

/**
 * @brief lwSocketDestroy Destroys LwSocketPair. May internally close socket subsystem.
 */
void lwSocketDestroy(LwSocketPair *pConnection);

/**
 * @brief lwSocketListen Start listening for client connection.
 * @param port Port to listen on
 * @return LW_OK on success
 *
 *  Creates server socket, configures & binds it and starts listening for incoming connections.
 */
LW_STATUS lwSocketListen(LwSocketPair *pConnection, int port);

/**
 * @brief lwSocketAccept Accepts client connection.
 * @return LW_OK on success
 */
LW_STATUS lwSocketAccept(LwSocketPair *pConnection, int timeout);

/**
 * @brief lwSocketCloseClient Closes client socket
 * @return LW_OK on success
 */
LW_STATUS lwSocketCloseClient(LwSocketPair *pConnection);

/**
 * @brief lwSocketCloseServer Closes server socket
 * @return LW_OK on success
 */
LW_STATUS lwSocketCloseServer(LwSocketPair *pConnection);

/**
 * @brief lwSocketRead Read data from client socket
 * @param pData Buffer for received data
 * @param n Size of buffer
 * @param block Whether the read should block or return immediately
 * @return -1 on error, amount of received data on success
 */
int lwSocketRead(LwSocketPair *pConnection, void *pData, size_t n, LwBool block);

/**
 * @brief lwSocketHasData Check if there is data on client socket
 * @return -1 on error, amount of data on client socket on success
 */
int lwSocketHasData(LwSocketPair *pConnection);

/**
 * @brief lwSocketWrite Transmit data to client
 * @param pData Buffer to be sent
 * @param n Size of buffer
 * @return -1 on error, number of bytes written on success
 */
int lwSocketWrite(LwSocketPair *pConnection, const void *pData, size_t n);

#endif
