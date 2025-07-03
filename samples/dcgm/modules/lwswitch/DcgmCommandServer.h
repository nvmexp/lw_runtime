
#pragma once

#include "LwcmThread.h"
#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>

/********************************************************************************/
/* Implements a socket server/client which act as a command interface to query  */
/* various states/information from FM during run-time                           */
/********************************************************************************/

class DcgmCommandClient;

/*********************************************************************************/
/* DcgmCommandServerCmds methods will be implemented by Global FM or Local FM  */
/* These commands can take sub options and GFM or LFM will have seperate commands*/
/*********************************************************************************/
class DcgmCommandServerCmds
{
public:
    DcgmCommandServerCmds() {;}
    virtual ~DcgmCommandServerCmds() {;}
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse) = 0;
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse) = 0;
};

/*********************************************************************************/
/* DcgmCommandServer implements a socket interface and accept connections. It als  */
/* parse the command string and calls the appropriate high level command handlers*/
/*********************************************************************************/
class DcgmCommandServer : public LwcmThread
{
public:

    DcgmCommandServer(DcgmCommandServerCmds *cmdHndlr,
                      int bindPort, char *sockPath);

    virtual ~DcgmCommandServer();

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
    static void onAccept(int fd, short ev, void *pThis);

    // helper function to clean-up when a command client connection close
    void onCommandClientClose();

    // helper function to dispatch actual commands
    void processCommand(std::string cmdLine);

    // helper function to write the command prompt
    void sendCommandPrompt();

    // helper function to write the command output to connected session
    void writeCommandOutput(std::string cmdResponse);

    // method to set socket as a non-blocking socket
    int setNonBlocking(int fd);

    struct event_base *mpEvbase; // Event base to receive connections and I/O events
    int mListenSockFd;           // Socket Descriptor for Server
    int mPortNumber;             // Port number for the server
    char mSocketPath[256];       // Socket path for all interface or no
    DcgmCommandClient *mpCommandClient;
    DcgmCommandServerCmds *mpCmdHndlr;
};

/*********************************************************************************/
/* DcgmCommandClient represents an accepted command client connection. Only one    */
/* instance of the DcgmCommandClient is created, ie only one socket connection can */
/* send commands at any time.                                                    */
/*********************************************************************************/
class DcgmCommandClient
{
    friend class DcgmCommandServer;

private:
    DcgmCommandClient(DcgmCommandServer *pCmdServer,
                      int clientFd, struct sockaddr_in remoteAddr);

    virtual ~DcgmCommandClient();

    int setOutputBuffer(std::string bufStr);
private:

    int mClientSockFd;
    struct sockaddr_in mRemoteSocketAddr;
    DcgmCommandServer *mpCmdServer;
    struct bufferevent *mpBufEv;       // The buffered event for this client.
    struct evbuffer *mpOutputBuffer;   // The output buffer for this client.
};

