/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <utils/lwassert.h>
#include <lwwatch-config.h>
#include <print.h>
#include <limits.h>

#include "lwsocket.h"
#include "lwsocket_porting.h"

#include "riscv_printing.h"

struct LwSocketPair
{
    LwInternalSocket client, server;
};

LwSocketPair *lwSocketCreate(void)
{
    LwSocketPair *pConnection = malloc(sizeof(LwSocketPair));
#if  LWWATCHCFG_IS_PLATFORM(WINDOWS)
    WSADATA wsaData;
#endif

    if (!pConnection)
        return NULL;

#if  LWWATCHCFG_IS_PLATFORM(WINDOWS)
    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0)
    {
        free(pConnection);
        return NULL;
    }
#endif

    memset(pConnection, 0, sizeof(LwSocketPair));
    pConnection->client = ILWALID_SOCKET;
    pConnection->server = ILWALID_SOCKET;

    return pConnection;
}

void lwSocketDestroy(LwSocketPair *pConnection)
{
    LW_ASSERT_OR_RETURN_VOID(pConnection);

    if (pConnection->client != ILWALID_SOCKET)
        lwSocketCloseClient(pConnection);
    if (pConnection->server != ILWALID_SOCKET)
        lwSocketCloseServer(pConnection);

#if  LWWATCHCFG_IS_PLATFORM(WINDOWS)
    WSACleanup();
#endif
    free(pConnection);
}

LW_STATUS lwSocketListen(LwSocketPair *pConnection, int port)
{
    struct sockaddr_in serverAddr;
    int socketOpt = 1;

    LW_ASSERT_OR_RETURN(pConnection != NULL, LW_ERR_GENERIC);
    LW_ASSERT_OR_RETURN(pConnection->server == ILWALID_SOCKET, LW_ERR_GENERIC);

    pConnection->server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (pConnection->server == ILWALID_SOCKET)
    {
        lprintf(PL_ERROR, "Error opening socket: %d.\n", SOCKET_FN_ERRNO);
        goto err;
    }
    /* Don't wait for TCP timeout of old server socket - reuse it. */
    if (setsockopt(pConnection->server, SOL_SOCKET, SO_REUSEADDR,
                   (const char *)&socketOpt, sizeof(int)) == SOCKET_ERROR)
    {
        lprintf(PL_ERROR, "Failed reconfiguring socket: %d.\n", SOCKET_FN_ERRNO);
        goto err_bind;
    }

    memset(&serverAddr, 0, sizeof(serverAddr));

    serverAddr.sin_family      = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port        = htons(port);
    if (bind(pConnection->server, (struct sockaddr *) &serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        lprintf(PL_ERROR, "Error binding socket: %d.\n", SOCKET_FN_ERRNO);
        goto err_bind;
    }

    if (listen(pConnection->server, 2) == SOCKET_ERROR)
    {
        lprintf(PL_ERROR, "Error listening: %d.\n", SOCKET_FN_ERRNO);
        goto err_listen;
    }

    lprintf(PL_DEBUG, "Listening on port %d\n", port);

    return LW_OK;

err_listen:
err_bind:
    SOCKET_FN_CLOSE(pConnection->server);
    pConnection->server = ILWALID_SOCKET;
err:
    return LW_ERR_GENERIC;
}


LW_STATUS lwSocketAccept(LwSocketPair *pConnection, int timeout)
{
    struct pollfd pfd;

    LW_ASSERT_OR_RETURN(pConnection != NULL, LW_ERR_GENERIC);
    LW_ASSERT_OR_RETURN(pConnection->server != ILWALID_SOCKET, LW_ERR_GENERIC);
    LW_ASSERT_OR_RETURN(pConnection->client == ILWALID_SOCKET, LW_ERR_GENERIC);

    lprintf(PL_DEBUG, "Waiting for client...\n");

    pfd.fd = pConnection->server;
    pfd.events = POLLIN;
    SOCKET_FN_POLL(&pfd, 1, timeout);

    if (!(pfd.revents & POLLIN)) {
        lprintf(PL_ERROR, "Timed out waiting for connection.\n");
        goto err_accept;
    }

    pConnection->client = accept(pConnection->server, NULL, NULL);
    if (pConnection->client == ILWALID_SOCKET)
    {
         lprintf(PL_ERROR, "Error accepting connection: %d.\n", SOCKET_FN_ERRNO);
         goto err_accept;
    }

    return LW_OK;

err_accept:
    return LW_ERR_GENERIC;
}

LW_STATUS lwSocketCloseClient(LwSocketPair *pConnection)
{
    LW_ASSERT_OR_RETURN(pConnection, LW_ERR_GENERIC);

    if (pConnection->client == ILWALID_SOCKET)
        return LW_OK;

    SOCKET_FN_CLOSE(pConnection->client);
    pConnection->client = ILWALID_SOCKET;

    return LW_OK;
}

LW_STATUS lwSocketCloseServer(LwSocketPair *pConnection)
{
    LW_ASSERT_OR_RETURN(pConnection, LW_ERR_GENERIC);

    if (pConnection ->server == ILWALID_SOCKET)
        return LW_OK;

    SOCKET_FN_CLOSE(pConnection->server);
    pConnection->server = ILWALID_SOCKET;

    return LW_OK;
}

int lwSocketRead(LwSocketPair *pConnection, void *pData, size_t n, LwBool block)
{
    int ret;
    struct pollfd pfd;

    LW_ASSERT_OR_RETURN(pConnection != NULL, -1);
    LW_ASSERT_OR_RETURN(pConnection->client != ILWALID_SOCKET, -1);
    LW_ASSERT_OR_RETURN(pData != NULL, -1);
    LW_ASSERT_OR_RETURN(n <= INT_MAX, -1);

    pfd.fd = pConnection->client;
    pfd.events = POLLIN;
    ret = SOCKET_FN_POLL(&pfd, 1, (block) ? -1 : 0);
    if (ret == 0)
        return -EAGAIN;

    // Double cast needed because Linux uses size_t and windows uses int
    ret = recv(pConnection->client, pData, (unsigned)(int)n, 0);
    if (ret == SOCKET_ERROR)
        return -1;
    return ret;
}

/* Warning - function not tested (yet) */
int lwSocketHasData(LwSocketPair *pConnection)
{
    int ret, bytesAv = 0;

    LW_ASSERT_OR_RETURN(pConnection != NULL, -1);

    ret = SOCKET_FN_IOCTL(pConnection->client, FIONREAD, &bytesAv);
    if (ret == SOCKET_ERROR)
    {
        lprintf(PL_ERROR, "ioctl() failed: %d.\n", SOCKET_FN_ERRNO);
        return -1;
    }
    return bytesAv;
}

// Send reply to gdb, client == cookie
int lwSocketWrite(LwSocketPair *pConnection, const void *pData, size_t n)
{
    int ret;

    LW_ASSERT_OR_RETURN(pConnection != NULL, -1);
    LW_ASSERT_OR_RETURN(pConnection->client != ILWALID_SOCKET, -1);
    LW_ASSERT_OR_RETURN(pData != NULL, -1);
    LW_ASSERT_OR_RETURN(n <= INT_MAX, -1);

    // Double cast needed because Linux uses size_t and windows uses int
    ret = send(pConnection->client, pData, (unsigned)(int)n, SOCKET_NO_SIGPIPE);
    if (ret == SOCKET_ERROR)
        return -1;
    return ret;
}
