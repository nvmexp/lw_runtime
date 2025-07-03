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

#pragma once

#include "FmThread.h"
#include <iostream>
#include <fstream>
#include <map>
#include <list>
#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#else
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif
#include "FMCommonTypes.h"
#include <util.h>

/********************************************************************************/
/* Implements a socket server/client which act as a command interface to query  */
/* various states/information from FM during run-time                           */
/********************************************************************************/

class FmCommandClient;

/*********************************************************************************/
/* FmCommandServerCmds methods will be implemented by Global FM or Local FM  */
/* These commands can take sub options and GFM or LFM will have seperate commands*/
/*********************************************************************************/
class FmCommandServerCmds
{
public:
    FmCommandServerCmds() {;}
    virtual ~FmCommandServerCmds() {;}
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse) = 0;
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse) = 0;
};

/*********************************************************************************/
/* FmCommandServer implements a socket interface and accept connections. It also */
/* parse the command string and calls the appropriate high level command handlers*/
/*********************************************************************************/
class FmCommandServer : public FmThread
{
public:

    FmCommandServer(FmCommandServerCmds *cmdHndlr,
                    int bindPort, char *sockPath);

    virtual ~FmCommandServer();

    // Implementing virtual method from the base class 
    void run(void);

    // used to start the listening socket, registers with libevent 
    // and listens for incoming connections
    int serve();

    // get the libeven event base pointer    
    struct event_base* getEventBase();

    // libevent callback when the socket is ready for read
    static void bufferReadCB(struct bufferevent *bev, void *pThis);

    // libevent callback when the socket is ready for write
    static void bufferWriteCB(struct bufferevent *bev, void *pThis);

    // libevent callback when the socket is has events like close, error etc
    static void bufferEventCB(struct bufferevent *bev, short events, void *pThis);

    static const std::string _mCmdPrompt;

private:

     // method is used to initialize the TCP Socket
    int initTCPSocket();

    // Callback registered with libevent to accept the incoming connection
    static void onAccept(evutil_socket_t fd, short ev, void *pThis);

    // helper function to clean-up when a command client connection close
    void onCommandClientClose();

    // helper function to dispatch actual commands
    void processCommand(std::string cmdLine);

    // helper function to write the command prompt
    void sendCommandPrompt();

    // helper function to write the command output to connected session
    void writeCommandOutput(std::string cmdResponse);

    // method to set socket as a non-blocking socket
    int setNonBlocking(evutil_socket_t fd);

    struct event_base *mpEvbase; // Event base to receive connections and I/O events
    evutil_socket_t mListenSockFd;           // Socket Descriptor for Server
    int mPortNumber;             // Port number for the server
    char mSocketPath[256];       // Socket path for all interface or no
    FmCommandClient *mpCommandClient;
    FmCommandServerCmds *mpCmdHndlr;
};

/*********************************************************************************/
/* FmCommandClient represents an accepted command client connection. Only one    */
/* instance of the FmCommandClient is created, ie only one socket connection can */
/* send commands at any time.                                                    */
/*********************************************************************************/
class FmCommandClient
{
    friend class FmCommandServer;

private:
    FmCommandClient(FmCommandServer *pCmdServer,
                    int clientFd, struct sockaddr_in remoteAddr);

    virtual ~FmCommandClient();

    int setOutputBuffer(std::string bufStr);
private:

    int mClientSockFd;
    struct sockaddr_in mRemoteSocketAddr;
    FmCommandServer *mpCmdServer;
    struct bufferevent *mpBufEv;       // The buffered event for this client.
    struct evbuffer *mpOutputBuffer;   // The output buffer for this client.
};

