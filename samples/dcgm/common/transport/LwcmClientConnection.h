/* 
 * File:   LwcmClientConnection.h
 */

#ifndef LWCMCLIENTCONNECTION_H
#define	LWCMCLIENTCONNECTION_H

#include "LwcmProtocol.h"
#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/tcp.h>
#include <string.h>
#include "LwcmThread.h"
#include "LwcmRequest.h"
#include "LwcmConnection.h"
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class LwcmClientListener : public LwcmThread {
public:
    /*****************************************************************************
     * Client Listener constructor to create event base
     *****************************************************************************/
    LwcmClientListener();

    /*****************************************************************************
     * Client Listener destructor
     *****************************************************************************/
    ~LwcmClientListener();

    /*****************************************************************************
     * Run method for listener thread
     *****************************************************************************/
    void run();
    
    /*****************************************************************************
     * This method is used to stop the client listener to receive any incoming
     * packet
     *****************************************************************************/
    void StopClientListener();

    /*****************************************************************************
     * This message is used to get event base
     *****************************************************************************/
    struct event_base * GetBase();
    
    /*****************************************************************************
     * This method serves as callback which is ilwoked by Libevent (Listener thread)
     * when there is a message to be read on the socket
     *****************************************************************************/
    static void ReadCB(struct bufferevent *bev, void *ctx);     

private:
    /*****************************************************************************
     * Lock and Unlock Methods used with condition
     *****************************************************************************/
    void Lock();
    void UnLock();
    
    /*****************************************************************************
     * Dummy Callback to keep Client listener alive
     *****************************************************************************/
    static void DummyCB(evutil_socket_t, short, void *);

    /*****************************************************************************
     * This method uses to read event buffers
     * This method provides thread safe way of using libevent bufferevents.
     *****************************************************************************/    
    static size_t ReadEventBuffer(struct bufferevent *bev, void *data, size_t size);
    
    /*****************************************************************************
     * This method uses to get length of event buffer.
     * This method provides thread safe way of using libevent bufferevents.
     *****************************************************************************/    
    static size_t GetEventBufferLength(struct bufferevent *bev);    
    struct event_base *mpBase;
};


class LwcmClientConnection : public LwcmConnection {
public:
     
    /*****************************************************************************
     * To establish connection with Host Engine
     *****************************************************************************/
    LwcmClientConnection(LwcmConnectionHandler *pConnHandler, LwcmClientListener *pClientBase,
                         char *identifier, int port_number, bool tryConnection, 
                         bool addressIsUnixSocket, int connectionTimeoutMs = 5000);
    
    /*****************************************************************************
     * Destroy Connection
     *****************************************************************************/
    virtual ~LwcmClientConnection();
    
    /*****************************************************************************
     * This method is used to send message to the HostEngine
     *****************************************************************************/    
    int SetOutputBuffer(LwcmMessage *pLwcmMsg);
    
    /*****************************************************************************
     * This method is used to disable any further notifications on the connection
     *****************************************************************************/
    void DisableConnectionNotifications();    

    /*****************************************************************************
     * This method is used to pass messages received with no corresponding active
       request id. This is to remove the req-resp semantics 
       default implementation will ignore the message
     *****************************************************************************/
    virtual void ProcessUnSolicitedMessage(LwcmMessage *msg) { };
    
    /*****************************************************************************
     * This method serves as a callback to receive events for the connection
     *****************************************************************************/
    static void EventCB(bufferevent* bev, short events, void *ptr);

    /*****************************************************************************
     * @brief This method is used to wait for the connection to succeed or fail
     *
     * @param milliseconds Amount of milliseconds to wait before returning a result.
     * @return:
     *      0  : On success;
     *      <0 : On Failure
     *
     *****************************************************************************/
    int WaitForConnection(unsigned int milliseconds);

    /*****************************************************************************
     * Disable notifications from libevent for this connection
     *
     * Return: Nothing
     * 
     *****************************************************************************/
    void DisableLibeventCallbacks(void);

    /*****************************************************************************
     * This method is a soft destructor and used to safely uninitialize this
     * object from anywhere in the code.
     *
     * Return: Nothing
     * 
     *****************************************************************************/
    void Cleanup();

    /*****************************************************************************
     * This method is used to propagate connection changes to derived class
     * The default implementation from LwcmConnection will do nothing
     *****************************************************************************/
    virtual void SignalConnStateChange(ConnectionState state);
   
protected:
    struct bufferevent *bev;
    bool mAddressIsUnixSocket;          /* Is this address a unix domain socket (true) or a normal unix 
                                           TCP/IP socket (false) */
    struct sockaddr_in m_serv_addr;    /* Unix socket address used for specifying connection details */
    struct sockaddr_un m_un_addr;      /* Unix domain socket address used for specifying connection details */
    struct evbuffer *mpInputBuffer;    /* The input buffer for this client. */
    struct evbuffer *mpOutputBuffer;   /* The output buffer for this client. */
};

#endif	/* LWCMCLIENTCONNECTION_H */
