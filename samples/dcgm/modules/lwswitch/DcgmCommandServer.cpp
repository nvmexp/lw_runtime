
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <err.h>
#include <event.h>
#include <thread.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "logging.h"
#include "LwcmSettings.h"
#include "DcgmCommandServer.h"

const std::string DcgmCommandServer::_mCmdPrompt = "\n===";

DcgmCommandServer::DcgmCommandServer(DcgmCommandServerCmds *cmdHndlr,
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

    (void)evthread_use_pthreads();
    if ((mpEvbase = event_base_new()) == NULL) {
        PRINT_ERROR("", "DcgmCmdServer: unable to create a socket accept event base");
        throw std::runtime_error("DcgmCmdServer: unable to create a socket accept event base");
    }
}

DcgmCommandServer::~DcgmCommandServer()
{
    if (mpEvbase) {
        event_base_loopexit(mpEvbase, NULL);
    }

    int st = StopAndWait(60000);
    if (st) {
        PRINT_WARNING("", "CmdServer: Killing thread that is still running.");
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
DcgmCommandServer::initTCPSocket()
{
    struct sockaddr_in listen_addr;
    int reuseaddr_on = 1;

    // create our listening socket.
    mListenSockFd = socket(AF_INET, SOCK_STREAM, 0);
    if (mListenSockFd < 0) {
        PRINT_ERROR("", "CmdServer: ocket creation failed");
        return -1;
    }

    if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on))) {
        PRINT_ERROR("%d", "CmdServer: setsockopt(SO_REUSEADDR) failed. errno %d", errno);
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    if(strlen(mSocketPath) > 0) {
        // colwert mSocketPath to a number in network byte order
        if(!inet_aton(mSocketPath, &listen_addr.sin_addr)) {
        PRINT_ERROR("%s", "CmdServer: Unable to colwert \"%s\" to a network address.", mSocketPath);
            close(mListenSockFd);
            mListenSockFd = -1;
            return -1;
        }
    }

    listen_addr.sin_port = htons(mPortNumber);
    if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
        PRINT_ERROR("%d %d %d", "CmdServer: bind failed. port %d, address %d, errno %d",
                    mPortNumber, listen_addr.sin_addr.s_addr, errno);
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    return 0;
}

int
DcgmCommandServer::serve()
{
    struct event ev_accept;
    int sock_init;

    mListenSockFd = -1;
    sock_init = initTCPSocket();

    if (sock_init) {
        PRINT_ERROR("", "CmdServer: socket initialization failed");
        if (mListenSockFd != -1) {
            close(mListenSockFd);
            mListenSockFd = -1;
        }
        return -1;
    }

    if (listen(mListenSockFd, 1) < 0) {
        PRINT_ERROR("", "CmdServer: listen failed");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    // set the socket to non-blocking, this is essential in event
    // based programming with libevent.
    if (setNonBlocking(mListenSockFd)) {
        PRINT_ERROR("", "CmdServer: failed to set server socket to non-blocking");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    // we now have a listening socket, we create a read event to
    // be notified when a client connects.
    event_set(&ev_accept, mListenSockFd, EV_READ|EV_PERSIST, DcgmCommandServer::onAccept, this);
    event_base_set(mpEvbase, &ev_accept);
    event_add(&ev_accept, NULL);

    event_base_dispatch(mpEvbase);

    if (mListenSockFd != -1) {
        close(mListenSockFd);
        mListenSockFd = -1;
    }

    event_del(&ev_accept);

    return 0;
}

struct event_base*
DcgmCommandServer::getEventBase()
{
    return mpEvbase;
}

void
DcgmCommandServer::run(void)
{
    int ret;
    ret = serve();
    if (0 != ret) {
        PRINT_ERROR("", "CmdServer: Failed to start command server");
    }
}

void
DcgmCommandServer::onAccept(int fd, short ev, void *pThis)
{
    int client_fd;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    DcgmCommandServer *pServer;

    client_fd = accept(fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        PRINT_ERROR("", "CmdServer: accept failed");
        return;
    }

    // set the client socket to non-blocking mode.
    pServer = reinterpret_cast<DcgmCommandServer*>(pThis);
    if (pServer->setNonBlocking(client_fd)) {
        PRINT_ERROR("", "CmdServer: failed to set client socket to non-blocking");
        close(client_fd);
        return;
    }

    if (pServer->mpCommandClient != NULL) {
        // we already have a connection. So close this one
        // another way is to stop the listen once we accept a connection
        close(client_fd);
        return;
    }

    pServer->mpCommandClient = new DcgmCommandClient(pServer, client_fd, client_addr);
}

int DcgmCommandServer::setNonBlocking(int fd)
{
    int flags;

    flags = fcntl(fd, F_GETFL);
    if (flags < 0) return flags;
    flags |= O_NONBLOCK;
    if (fcntl(fd, F_SETFL, flags) < 0) return -1;
    return 0;
}

void
DcgmCommandServer::bufferReadCB(struct bufferevent *bev, void *pThis)
{
    DcgmCommandServer *pCmdServer = reinterpret_cast<DcgmCommandServer*>(pThis);
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
DcgmCommandServer::bufferWriteCB(struct bufferevent *bev, void *pThis)
{
    // do nothing
}

void
DcgmCommandServer::bufferEventCB(struct bufferevent *bev, short events, void *pThis)
{
    DcgmCommandServer *pCmdServer = reinterpret_cast<DcgmCommandServer*>(pThis);

    if (events & (BEV_EVENT_ERROR|BEV_EVENT_EOF)) {
        pCmdServer->onCommandClientClose();
    }
}

void
DcgmCommandServer::onCommandClientClose()
{
    if (mpCommandClient) {
        delete mpCommandClient;
        mpCommandClient = NULL;
    }
}

void
DcgmCommandServer::processCommand(std::string cmdLine)
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
DcgmCommandServer::writeCommandOutput(std::string cmdResponse)
{
    mpCommandClient->setOutputBuffer(cmdResponse);
}

void
DcgmCommandServer::sendCommandPrompt()
{
    mpCommandClient->setOutputBuffer(DcgmCommandServer::_mCmdPrompt);
}

DcgmCommandClient::DcgmCommandClient(DcgmCommandServer *pCmdServer, 
                                     int clientFd, struct sockaddr_in remoteAddr)
{
    mClientSockFd = clientFd;
    bcopy((char *)&remoteAddr, (char *)&mRemoteSocketAddr, sizeof(mRemoteSocketAddr));
    mpCmdServer = pCmdServer;

    if ((mpOutputBuffer = evbuffer_new()) == NULL) {
        throw std::runtime_error("CmdServer: failed to create output event buffer");
    }

    mpBufEv = bufferevent_socket_new(mpCmdServer->getEventBase(), mClientSockFd,
                                     BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    if (NULL == mpBufEv) {
        throw std::runtime_error("CmdServer: failed to create buffer event");
    }

    bufferevent_setcb(mpBufEv, DcgmCommandServer::bufferReadCB, DcgmCommandServer::bufferWriteCB,
                                DcgmCommandServer::bufferEventCB, mpCmdServer);
    bufferevent_enable(mpBufEv, EV_READ|EV_WRITE);

    // set the command prompt
    setOutputBuffer(DcgmCommandServer::_mCmdPrompt);
}

DcgmCommandClient::~DcgmCommandClient()
{
    if (mClientSockFd > 0) {
        close(mClientSockFd);
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
DcgmCommandClient::setOutputBuffer(std::string bufStr)
{
    bufferevent_lock(mpBufEv);
    evbuffer_add(mpOutputBuffer, bufStr.c_str(), bufStr.size());

    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        PRINT_ERROR("", "CmdServer: Failed to write message to event buffer");
        bufferevent_unlock(mpBufEv);
        return -2;
    }

    bufferevent_unlock(mpBufEv);
    return 0;    
}
