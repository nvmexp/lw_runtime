/* 
 * File:   FmServer.h
 */

#ifndef FMSERVER_H
#define	FMSERVER_H

#include "workqueue.h"
#include "FmThread.h"
#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <event2/event.h>
#include <event2/event_struct.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <event2/event_compat.h>
#include "FmRequest.h"
#include "FmConnection.h"

#define FM_NUM_SERVER_WORKER_THREADS 2
#define FM_SERVER_CONNECTION_BACKLOG 6
#define FM_ENABLE_LIBEVENT_LOGGING   1 

class FmServerConnection;

using namespace std;

/* todo: 
 * 1) Check if timeout is needed for each connection
 */

enum ServerState {
    FM_SERVER_NOT_STARTED = 0,
    FM_SERVER_RUNNING,
    FM_SERVER_STOPPED,
};

class FmSocket : public FmThread {
public:

    /**
     * Parameterized Constructor for FM Socket class
     *
     * port       IN: TCP port to listen on. This is only used if isTcp == 1.
     * sockpath   IN: This is the path passed to bind() when creating the socket
     *                For isTcp == 1, this is the bind address. "" or NULL = All interfaces
     *                For isTcp == 0, this is the path to the domain socket to use
     * isTCP      IN: Whether to listen on a TCP/IP socket (1) or a unix domain socket (0)
     * numWorkers IN: Number of worker threads to drain socket messages. Default is 2
     */
    FmSocket(int port, char *sockpath, int isTCP, int numWorkers = FM_NUM_SERVER_WORKER_THREADS);
    
    /**
     * Destructor for FM server class
     */
    virtual ~FmSocket();

    /**
     * Implementing virtual method from the base class 
     */
    void run(void);
    
    /**
     * This method is used to initialize the TCP Socket
     */
    int InitTCPSocket();
    
    /**
     * This method is used to initialize the Unix Socket
     */
    int InitUnixSocket();
    
    /**
     * This method is used to start the listening socket, registers with libevent 
     * and listens for incoming connections
     */
    int Serve();
    
    /**
     * This method is used to gracefully stop the server from listening to any
     * new connections or requests
     */
    void StopServer();
    
    /**
     * This method is used to wait for server to get ready in order to start 
     * accepting connections/packets from the clients
     */
    int WaitToStart();
    
    /**
     * The class extending the server class must implement this method to 
     * handle incoming requests
     */
    virtual int OnRequest(fm_request_id_t requestId, FmServerConnection *pConnection) = 0;

    /**
     * Notify the server object when the corresponding listener accepted a server connection
     * and added it to the connection handler
     * default implementation do nothing
     */
    virtual void OnConnectionAdd(fm_connection_id_t connectionId, FmServerConnection *pConnection) { };

    /**
     * Notify the server object when an accepted connection is closed and is removed from connection handler
     * default implementation do nothing
     */
    virtual void OnConnectionRemove(fm_connection_id_t connectionId, FmServerConnection *pConnection) { };

    
    /**
     * The FmServerConnection class uses this method to add the connection to
     * a worker queue
     */
    void AddRequestToQueue(FmServerConnection *pConnection, void *args);
    
    /*****************************************************************************
     * This method is used to get reference to connection handler
     *****************************************************************************/
    FmConnectionHandler *GetConnectionHandler();
    
    /**
     * This method returns the event base maintained by the server
     * @return 
     */
    struct event_base * GetEventBase();
    
    /**
     * Callback Ilwoked from the worker queue when thread is ready to process the request
     */
    static void ProcessRequest(job_t *pJob);
    

    
    /*****************************************************************************
     * Callback registered with libevent. This method is called when data has 
     * been read from the socket and is available to the application.
     *****************************************************************************/
    static void BufferReadCB(struct bufferevent *bev, void *_pThis);
    
    /*****************************************************************************
     * This method is called when the write buffer has reached a low watermark.
	 * That usually means that when the write buffer is 0 length,
	 * this callback will be called.  It must be defined, but you
	 * don't actually have to do anything in this callback.
     *****************************************************************************/
    static void BufferWriteCB(struct bufferevent *bev, void *_pThis);
    
    /*****************************************************************************
     * This method is called when there is a socket error.  This is used to 
     * detect that the client disconnected or other socket errors.
     *****************************************************************************/
    static void BufferEventCB(struct bufferevent *bev, short what, void *_pThis);    
    
    
private:
    /*****************************************************************************
     * Lock and Unlock Methods used with condition
     *****************************************************************************/
    void Lock();
    void UnLock();    
    
    /* Callback registered with libevent to accept the incoming connection */
    static void OnAccept(evutil_socket_t fd, short ev, void *_pThis);
    static void RecordDebugCB(int severity, const char *msg);
    static void DiscardDebugCB(int severity, const char *msg);
    
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
    
    /* This method is used to set socket as a non-blocking socket */
    int SetNonBlocking(evutil_socket_t fd);
    struct event_base *mpEvbase;                    /* Event base to receive connections and I/O events */
    workqueue_t mWorkQueue;                         /* Multi-threaded work queue */
    int mNumWorkers;                                /* Number of worker threads */
    evutil_socket_t mListenSockFd;                              /* Socket Descriptor for Server */
    int mPortNumber;                                /* Port number for the server */
    int mIsConnectionTCP;                           /* Flag indicating whether connection is via a TCP port */
    char mSocketPath[256];                          /* Socket path in case of a Unix socket */
    static ofstream mDebugLogFile;                  /* Debug mode only: Debug file for maintaining libvent logs and warnings */
    FmConnectionHandler *mpConnectionHandler;     /* Maintains list of connections */
    LWOSCriticalSection mMutex;                     /* Mutex for the condition variable */
    lwosCV mCondition;                              /* Condition used to signal when the server stopped listening for connections */
    ServerState mServerRunState;                    /* Flag to identify server run state */
};

/**
 * Structure for user data to push to worker queue
 */
typedef struct FmRequestInfo
{
    fm_request_id_t requestId;        /* Represents Request ID (Unique for a connection) */
    FmServerConnection *pConnection;  /* Represents Server Connection */
} FmRequestInfo_t;


class FmServerConnection : public FmConnection
{
public:
    
    /*****************************************************************************
     * Constructor to manage new connection on the server. It creates
     * the buffered event and registers necessary read, write and error 
     * callbacks with libevent
     *****************************************************************************/
    FmServerConnection(FmConnectionHandler *pConnHandler,
                       FmSocket *pServer, int fd, struct sockaddr_in remoteAddr);

    /*****************************************************************************
     * Destructor for connection
     *****************************************************************************/
    virtual ~FmServerConnection();
    
    /*****************************************************************************
     * This method is used to Set the Output Buffer to be sent back to the 
     * client
     *****************************************************************************/
    int SetOutputBuffer(FmSocketMessage *pMsg);
    
    /*****************************************************************************
     * This method is used to disable any further notifications on the connection
     *****************************************************************************/
    void DisableConnectionNotifications();        
    
    /*****************************************************************************
     * This method is used to get FM Server Instance
     *****************************************************************************/
    FmSocket *GetServer();

    struct sockaddr_in GetRemoteSocketAddr();

    /*****************************************************************************
     * This method is used to enable or disable a client's watches persisting
     * after the client disconnects. 
     * true = persist a client's watches after disconnect.
     * false = clean up everything this connection was watching when it goes away
     *****************************************************************************/
    void SetPersistAfterDisconnect(bool value);
    
    /*****************************************************************************
     * Get the value set by SetPersistAfterDisconnect()
     */
    bool GetPersistAfterDisconnect();
    
private:
    
    struct sockaddr_in mRemoteSocketAddr;

    FmSocket *mpFmServerInstance;  /* Instance for FM Server */
	struct bufferevent *mpBufEv; 	   /* The buffered event for this client. */
    struct evbuffer *mpInputBuffer;    /* The input buffer for this client. */
    struct evbuffer *mpOutputBuffer;   /* The output buffer for this client. */
    int mFd;                           /* The client's socket. */

    bool m_persistAfterDisconnect;     /* Whether or not to persist watches..etc created
                                          by this connection after disconnect. 
                                          true = persist
                                          false = clean up after this connection */
};

#endif	/* FMSERVER_H */
