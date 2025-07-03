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
#include <sys/types.h>
#ifdef __linux__
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <err.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>

#include <event.h>
#include <util.h>
#include <buffer.h>
#include <buffer_compat.h>
#include <bufferevent.h>
#include <bufferevent_struct.h>
#include <event_struct.h>
#include <event_compat.h>
#include <thread.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "fm_log.h"
#include "FMCommandServer.h"

const std::string FmCommandServer::_mCmdPrompt = "\n===";

FmCommandServer::FmCommandServer(FmCommandServerCmds *cmdHndlr,
                                 int bindPort, char *sockPath)
{
    mpCmdHndlr = cmdHndlr;
    mPortNumber = bindPort;
    memset(mSocketPath, 0, sizeof(mSocketPath));
    if(sockPath) {
        strncpy(mSocketPath, sockPath, sizeof(mSocketPath)-1);
    }

    mpEvbase = NULL;
    mListenSockFd = -1;
    mpCommandClient = NULL;
#ifdef __linux__
    (void)evthread_use_pthreads();
#else
    evthread_use_windows_threads();
#endif
    if ((mpEvbase = event_base_new()) == NULL) {
        FM_LOG_ERROR("command server: unable to allocate event base object for socket");
        throw std::runtime_error("command server: unable to allocate event base object for socket");
    }
}

FmCommandServer::~FmCommandServer()
{
    Stop();
    
    if (mpEvbase) {
        event_base_loopexit(mpEvbase, NULL);
    }

    int st = StopAndWait(60000);
    if (st) {
        FM_LOG_WARNING("command server: killing socket listener thread after stop request timeout");
        Kill();
    }

    /* Wait until after our worker is gone to actually free the event base as we can otherwise
       cause a use-after-free crash in mpEvbase */
    if (mpEvbase) {
        event_base_free(mpEvbase);
        mpEvbase = NULL;
    }

}

int
FmCommandServer::initTCPSocket()
{
    struct sockaddr_in listen_addr;
    int reuseaddr_on = 1;

    // create our listening socket.
    mListenSockFd = socket(AF_INET, SOCK_STREAM, 0);
    if (mListenSockFd < 0) {
        FM_LOG_ERROR("command server: socket object creation failed");
        return -1;
    }

    if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuseaddr_on, sizeof(reuseaddr_on)))
    {
        FM_LOG_ERROR("command server: failed to set socket property (SO_REUSEADDR). errno %d", errno);
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    if(strlen(mSocketPath) > 0) {
        // colwert mSocketPath to a number in network byte order
        if(!inet_aton(mSocketPath, &listen_addr.sin_addr)) {
            FM_LOG_ERROR("command server: unable to colwert provided socket path \"%s\" to a network address.", mSocketPath);
            evutil_closesocket(mListenSockFd);
            mListenSockFd = -1;
            return -1;
        }
    }

    listen_addr.sin_port = htons(mPortNumber);
    if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
		char *ipStr = inet_ntoa(listen_addr.sin_addr);
        FM_LOG_ERROR("command server: socket bind operation failed for address %s port %d, with errno %d",
                     ipStr, mPortNumber, errno);
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    return 0;
}

int
FmCommandServer::serve()
{
    struct event ev_accept;
    int sock_init;

    mListenSockFd = -1;
    sock_init = initTCPSocket();

    if (sock_init) {
        FM_LOG_ERROR("command server: socket initialization request failed");
        if (mListenSockFd != -1) {
            evutil_closesocket(mListenSockFd);
            mListenSockFd = -1;
        }
        return -1;
    }

    if (listen(mListenSockFd, 1) < 0) {
        FM_LOG_ERROR("command server: socket listen request failed");
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    // set the socket to non-blocking, this is essential in event
    // based programming with libevent.
    if (setNonBlocking(mListenSockFd)) {
        FM_LOG_ERROR("command server: failed to set server socket non-blocking property");
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    // we now have a listening socket, we create a read event to
    // be notified when a client connects.
    event_set(&ev_accept, mListenSockFd, EV_READ|EV_PERSIST, FmCommandServer::onAccept, this);
    event_base_set(mpEvbase, &ev_accept);
    event_add(&ev_accept, NULL);

#ifdef SIM_BUILD // MODS SIM
    while (!ShouldStop()) {
        event_base_loop(mpEvbase, EVLOOP_NONBLOCK);
        /* Ensure that we don't deadlock while running in simulation */
        lwosThreadYield();
    }
#else
    event_base_dispatch(mpEvbase);
#endif

    if (mListenSockFd != -1) {
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
    }

    event_del(&ev_accept);

    return 0;
}

struct event_base*
FmCommandServer::getEventBase()
{
    return mpEvbase;
}

void
FmCommandServer::run(void)
{
    int ret;
    ret = serve();
    if (0 != ret) {
        FM_LOG_ERROR("command server: failed to start command server interface");
    }
}

void
FmCommandServer::onAccept(evutil_socket_t fd, short ev, void *pThis)
{
    int client_fd;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    FmCommandServer *pServer;

    client_fd = accept(fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        FM_LOG_ERROR("command server: socket accept operation failed");
        return;
    }

    // set the client socket to non-blocking mode.
    pServer = reinterpret_cast<FmCommandServer*>(pThis);
    if (pServer->setNonBlocking(client_fd)) {
        FM_LOG_ERROR("command server: failed to set client socket non-blocking property");
        evutil_closesocket(client_fd);
        return;
    }

    if (pServer->mpCommandClient != NULL) {
        // we already have a connection. So close this one
        // another way is to stop the listen once we accept a connection
        evutil_closesocket(client_fd);
        return;
    }

    pServer->mpCommandClient = new FmCommandClient(pServer, client_fd, client_addr);
}

int FmCommandServer::setNonBlocking(evutil_socket_t fd)
{
    evutil_make_socket_nonblocking(fd);
    return 0;
}

void
FmCommandServer::bufferReadCB(struct bufferevent *bev, void *pThis)
{
    FmCommandServer *pCmdServer = reinterpret_cast<FmCommandServer*>(pThis);
    char *cmdLine;
    size_t cmdLen;

    cmdLine = evbuffer_readline(bev->input);
    if(cmdLine == NULL) {
        // No data, or data has arrived, but no end-of-line was found
        return;
    }
    std::string strCmd(cmdLine);
    pCmdServer->processCommand(strCmd);
    free(cmdLine);
}

void
FmCommandServer::bufferWriteCB(struct bufferevent *bev, void *pThis)
{
    // do nothing
}

void
FmCommandServer::bufferEventCB(struct bufferevent *bev, short events, void *pThis)
{
    FmCommandServer *pCmdServer = reinterpret_cast<FmCommandServer*>(pThis);

    if (events & (BEV_EVENT_ERROR|BEV_EVENT_EOF)) {
        pCmdServer->onCommandClientClose();
    }
}

void
FmCommandServer::onCommandClientClose()
{
    if (mpCommandClient) {
        delete mpCommandClient;
        mpCommandClient = NULL;
    }
}

void
FmCommandServer::processCommand(std::string cmdLine)
{
    std::string cmdResponse;
    std::string whitespace(" \t");
    size_t cmdLength = cmdLine.find_first_not_of(whitespace);

    if (cmdLength == std::string::npos) {
        // the line was empty ie no command was given
        sendCommandPrompt();
        return;
    }

    std::string quitCmd("/quit");
    if (!cmdLine.compare(0, quitCmd.size(), quitCmd)) {
        onCommandClientClose();
        return;
    } 

    std::string rumCmd("/run");
    if (!cmdLine.compare(0, rumCmd.size(), rumCmd)) {
        mpCmdHndlr->handleRunCmd(cmdLine, cmdResponse);
    } 

    std::string queryCmd("/query");
    if (!cmdLine.compare(0, queryCmd.size(), queryCmd)) {
        mpCmdHndlr->handleQueryCmd(cmdLine, cmdResponse);
    }

    // write the output and the prompt
    if (cmdResponse.size()) {
        writeCommandOutput(cmdResponse);
    }

    // send the prompt again
    sendCommandPrompt();
}

void
FmCommandServer::writeCommandOutput(std::string cmdResponse)
{
    mpCommandClient->setOutputBuffer(cmdResponse);
}

void
FmCommandServer::sendCommandPrompt()
{
    mpCommandClient->setOutputBuffer(FmCommandServer::_mCmdPrompt);
}

FmCommandClient::FmCommandClient(FmCommandServer *pCmdServer, 
                                 int clientFd, struct sockaddr_in remoteAddr)
{
    mClientSockFd = clientFd;
    bcopy((char *)&remoteAddr, (char *)&mRemoteSocketAddr, sizeof(mRemoteSocketAddr));
    mpCmdServer = pCmdServer;

    if ((mpOutputBuffer = evbuffer_new()) == NULL) {
        throw std::runtime_error("command server: failed to allocate event buffer for socket");
    }

    mpBufEv = bufferevent_socket_new(mpCmdServer->getEventBase(), mClientSockFd,
                                     BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    if (NULL == mpBufEv) {
        throw std::runtime_error("command server: failed to set socket listening events through buffer event");
    }

    bufferevent_setcb(mpBufEv, FmCommandServer::bufferReadCB, FmCommandServer::bufferWriteCB,
                                FmCommandServer::bufferEventCB, mpCmdServer);
    bufferevent_enable(mpBufEv, EV_READ|EV_WRITE);

    // set the command prompt
    setOutputBuffer(FmCommandServer::_mCmdPrompt);
}

FmCommandClient::~FmCommandClient()
{
    if (mClientSockFd > 0) {
        evutil_closesocket(mClientSockFd);
        mClientSockFd = -1;
    }

    if (mpBufEv != NULL) {
        bufferevent_free(mpBufEv);
        mpBufEv = NULL;
    }

    if (mpOutputBuffer != NULL) {
        evbuffer_free(mpOutputBuffer);
        mpOutputBuffer = NULL;
    }
}

int
FmCommandClient::setOutputBuffer(std::string bufStr)
{
    bufferevent_lock(mpBufEv);
    evbuffer_add(mpOutputBuffer, bufStr.c_str(), bufStr.size());

    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        FM_LOG_ERROR("command server: failed to write message to event buffer for sending over socket interface");
        bufferevent_unlock(mpBufEv);
        return -2;
    }

    bufferevent_unlock(mpBufEv);
    return 0;    
}
