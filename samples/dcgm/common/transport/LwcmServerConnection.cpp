/*
 * File:   LwcmServerConnection.cpp
 */

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
#include <signal.h>
#include "workqueue.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "logging.h"
#include "LwcmProtocol.h"
#include "dcgm_structs.h"
#include "LwcmServerRequest.h"
#include "LwcmServerConnection.h"
#include "LwcmSettings.h"

using namespace std;

ofstream LwcmServer::mDebugLogFile;

/*****************************************************************************
 * Constructor
 *****************************************************************************/
LwcmServer::LwcmServer(int port, char *sockpath, int isTCP, int numWorkers) {
    mPortNumber = port;
    mIsConnectionTCP = isTCP;
    mNumWorkers = numWorkers;

    memset(mSocketPath, 0, sizeof(mSocketPath));
    if(sockpath)
        strncpy(mSocketPath, sockpath, sizeof(mSocketPath)-1);

    mpEvbase = NULL;
    mListenSockFd = -1;
    mServerRunState = DCGM_SERVER_NOT_STARTED;
	lwosInitializeCriticalSection(&mMutex);
    lwosCondCreate(&mCondition);

    (void)evthread_use_pthreads();
	if ((mpEvbase = event_base_new()) == NULL) {
        PRINT_ERROR("", "ERROR: unable to create socket accept event base");
        DEBUG_STDERR("ERROR: unable to create socket accept event base");
        close(mListenSockFd);
        mListenSockFd = -1;
        throw std::runtime_error("ERROR: unable to create socket accept event base");
	}    

    mpConnectionHandler = new LwcmConnectionHandler();
}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
LwcmServer::~LwcmServer() {
    
    /* Since this is the last object to be deleted by the Host Engine. 
     * Other modules must have removed any references to the pending connection 
     * entries, so it's safe to decrement the count for all the connection entries
     * still present in the list. Ilwoke destructor of connection handler 
     * to decrement reference counts for all the pending connections (if any) */
    if (mpConnectionHandler)
        delete mpConnectionHandler;    

	lwosDeleteCriticalSection(&mMutex);
    lwosCondDestroy(&mCondition);    
    
    if (mpEvbase) {
        event_base_free(mpEvbase);
        mpEvbase = NULL;
    }        
    
#ifdef _DEBUG
    if (mDebugLogFile.is_open()) {
        mDebugLogFile.close();
    }
#endif
    
    int st = StopAndWait(60000);
    if (st) {
        PRINT_WARNING("", "Killing server thread that is still running.");
        DEBUG_STDERR("Killing server thread that is still running.");
        Kill();
    }    
}

/*****************************************************************************
 * This method is used to record libevent debug/warning events to a file
 * The message is added for debugging purpose.
 *****************************************************************************/
void LwcmServer::RecordDebugCB(int severity, const char *msg)
{
    const char *s;

    if (!mDebugLogFile.is_open())
        return;

    switch (severity) {
        case _EVENT_LOG_DEBUG: s = "debug"; break;
        case _EVENT_LOG_MSG:   s = "msg";   break;
        case _EVENT_LOG_WARN:  s = "warn";  break;
        case _EVENT_LOG_ERR:   s = "error"; break;
        default:               s = "?";     break; /* never reached */
    }

    mDebugLogFile << s << ":" << msg << endl;
}

/*****************************************************************************
 * This method is used to discard the debugs/earnings
 *****************************************************************************/
void LwcmServer::DiscardDebugCB(int severity, const char *msg)
{
    // Do Nothing
}

/*****************************************************************************
 * This method is used to initialize a TCP socket
 *****************************************************************************/
int LwcmServer::InitTCPSocket()
{
	struct sockaddr_in listen_addr;
	int reuseaddr_on;

	/* Create our listening socket. */
	mListenSockFd = socket(AF_INET, SOCK_STREAM, 0);
	if (mListenSockFd < 0) {
        std::cerr << "ERROR: socket creation failed" << std::endl;
        return -1;
	}

#ifdef _DEBUG
    /* Register to log the warnings/error to a file */
    if (DCGM_ENABLE_LIBEVENT_LOGGING) {
        char *fileName = (char *)"/tmp/libevent-logs";
        unlink(fileName);
        mDebugLogFile.open(fileName);
        if (!mDebugLogFile.is_open()) {
            DEBUG_STDERR("ERROR: Debug file creation failed");
            PRINT_ERROR("", "ERROR:  Debug file creation failed. Will not log libevent");
            event_set_log_callback(DiscardDebugCB);
        }
        else
            event_set_log_callback(RecordDebugCB);
    } else {
        event_set_log_callback(DiscardDebugCB);
    }
#endif

	reuseaddr_on = 1;
	if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on)))
    {
	    PRINT_ERROR("%d", "ERROR: setsockopt(SO_REUSEADDR) failed. errno %d", errno);
	    std::cerr << "ERROR: setsockopt(SO_REUSEADDR) failed. errno " << errno << std::endl;
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

	memset(&listen_addr, 0, sizeof(listen_addr));
	listen_addr.sin_family = AF_INET;

	listen_addr.sin_addr.s_addr = INADDR_ANY;
	if(strlen(mSocketPath) > 0)
	{
	    /* Colwert mSocketPath to a number in network byte order */
	    if(!inet_aton(mSocketPath, &listen_addr.sin_addr))
	    {
	        PRINT_ERROR("%s", "ERROR: Unable to colwert \"%s\" to a network address.", mSocketPath);
	        std::cerr << "ERROR: Unable to colwert \"" << mSocketPath << "\" to a network address." << std::endl;
	        close(mListenSockFd);
            mListenSockFd = -1;
            return -1;
	    }
	}

	listen_addr.sin_port = htons(mPortNumber);
	if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
        PRINT_ERROR("%d %d %d", "ERROR: bind failed. port %d, address %d, errno %d",
                    mPortNumber, listen_addr.sin_addr.s_addr, errno);
        std::cerr << "ERROR: TCP bind failed for port " << mPortNumber << " address "
                  << listen_addr.sin_addr.s_addr << " errno " << errno << std::endl;
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
	}

    return 0;
}

/*****************************************************************************
 * This method is used to initialize a Unix domain socket
 *****************************************************************************/
int LwcmServer::InitUnixSocket()
{
	struct sockaddr_un listen_addr;
	int reuseaddr_on;

	/* Create our listening socket. */
	mListenSockFd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (mListenSockFd < 0) {
        PRINT_ERROR("", "ERROR: socket creation failed");
        return -1;
	}

#ifdef _DEBUG
    /* Register to log the warnings/error to a file */
    if (DCGM_ENABLE_LIBEVENT_LOGGING) {
        char *fileName = (char *)"/tmp/libevent-logs";
        unlink(fileName);
        mDebugLogFile.open(fileName);
        if (!mDebugLogFile.is_open()) {
            DEBUG_STDERR("ERROR: Debug file creation failed");
            PRINT_ERROR("", "ERROR:  Debug file creation failed. Will not log libevent");
            event_set_log_callback(DiscardDebugCB);
        }
        else
            event_set_log_callback(RecordDebugCB);
    } else {
        event_set_log_callback(DiscardDebugCB);
    }
#endif

	reuseaddr_on = 1;
	if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on)))
    {
        DEBUG_STDERR("ERROR: set socket option failed");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

	memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sun_family = AF_UNIX;
    strncpy(listen_addr.sun_path, mSocketPath, sizeof(listen_addr.sun_path)-1);
    unlink(mSocketPath); /* Make sure the path doesn't exist or bind will fail */
    if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
        PRINT_ERROR("", "ERROR: bind failed");
        DEBUG_STDERR("ERROR: bind failed");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

    return 0;
}

/*****************************************************************************
 This method is used to start the listening socket, registers with libevent
 and listens for incoming connections
 *****************************************************************************/
int LwcmServer::Serve()
{
	struct event ev_accept;
    int sock_init;

    mListenSockFd = -1;

    if (mIsConnectionTCP) {
        sock_init = InitTCPSocket();
    } else {
        sock_init = InitUnixSocket();
    }
    if (sock_init) {
        PRINT_ERROR("", "ERROR: socket initialization failed");
        DEBUG_STDERR("ERROR: socket initialization failed");
        if (mListenSockFd != -1) {
            close(mListenSockFd);
            mListenSockFd = -1;
        }
        return -1;
    }

	if (listen(mListenSockFd, DCGM_SERVER_CONNECTION_BACKLOG) < 0) {
        PRINT_ERROR("", "ERROR: listen failed");
        DEBUG_STDERR("ERROR: listen failed");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
	}

    /* Set the socket to non-blocking, this is essential in event
     * based programming with libevent. */
    if (this->SetNonBlocking(mListenSockFd)) {
        PRINT_ERROR("", "ERROR: failed to set server socket to non-blocking");
        DEBUG_STDERR("ERROR: failed to set server socket to non-blocking");
        close(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

	/* Initialize work queue. */
	if (workqueue_init(&mWorkQueue, mNumWorkers)) {
        PRINT_ERROR("", "ERROR: failed to create work queue");
        DEBUG_STDERR("ERROR: failed to create work queue");
		close(mListenSockFd);
        mListenSockFd = -1;
		return -1;
	}

	/* We now have a listening socket, we create a read event to
	 * be notified when a client connects. */
	// event_set(&ev_accept, listenfd, EV_READ|EV_PERSIST, lwcmServerOnAccept, (void *)&workqueue);
    event_set(&ev_accept, mListenSockFd, EV_READ|EV_PERSIST, LwcmServer::OnAccept, this);
	event_base_set(mpEvbase, &ev_accept);
	event_add(&ev_accept, NULL);

    PRINT_DEBUG("", "Host Engine Started");
    DEBUG_STDOUT("Host Engine Listener Started");

    /* Signal the waiting thread to notify that the server is running to receive connections */
    Lock();
    mServerRunState = DCGM_SERVER_RUNNING;
    lwosCondSignal(&mCondition);
    UnLock();

    event_base_dispatch(mpEvbase);

    DEBUG_STDOUT("Host Engine Socket Listener Stopped");

    if (mListenSockFd != -1) {
        close(mListenSockFd);
        mListenSockFd = -1;
    }

    event_del(&ev_accept);

    /* Signal that server has stopped and will not receive/process any packets */
    Lock();
    mServerRunState = DCGM_SERVER_STOPPED;
    lwosCondSignal(&mCondition);
    UnLock();

    /* Shut down the work queue last */
    workqueue_shutdown(&mWorkQueue);
	return 0;
}

/*****************************************************************************
 * Implements run method extended from LwcmThread
 *****************************************************************************/
void LwcmServer::run(void)
{
    int ret;
    ret = Serve();
    if (0 != ret) {
        cout << "Failed to start host engine server" << endl;
    }
}


/*****************************************************************************/
int LwcmServer::WaitToStart()
{
    unsigned int timeout = 1000; /* 1 seconds timeout */
    int ret;

    /* Wait for timeout to get the server started. Return an error if server
       can't be started */
    Lock();
    while (mServerRunState == DCGM_SERVER_NOT_STARTED) {
        ret = lwosCondWait(&mCondition, &mMutex, timeout);
        if ((LWOS_TIMEOUT == ret) && (HasRun()) && (HasExited())) {
            UnLock();
            return -1;
        }
    }
    UnLock();
    
    return 0;
}

/*****************************************************************************
 * This method is used to gracefully exit the server
 *****************************************************************************/
void LwcmServer::StopServer()
{
    unsigned int timeout = 1000; /* 1 seconds timeout */

    /* Wait for server to be stopped */
    Lock();
    if (mServerRunState != DCGM_SERVER_RUNNING) {
        /* The server instance may not have started or already stopped.
           Don't attempt to break the eventloop in this case as it 
           may get stuck */
        UnLock();
        return;
    }

    /* exit event loop and wait for server thread to stop */
    event_base_loopexit(mpEvbase, NULL); 
    while (mServerRunState != DCGM_SERVER_STOPPED)
        (void)lwosCondWait(&mCondition, &mMutex, timeout);
    UnLock();
}

/*****************************************************************************
  Callback registered with libevent to accept the incoming connection
 *****************************************************************************/
void LwcmServer::OnAccept(int fd, short ev, void *_pThis)
{
	int client_fd;
	struct sockaddr_in client_addr;
	socklen_t client_len = sizeof(client_addr);
    LwcmConnection* pConnection = NULL;
    LwcmServer * pServer;

	client_fd = accept(fd, (struct sockaddr *)&client_addr, &client_len);
	if (client_fd < 0) {
        PRINT_ERROR("", "ERROR: accept failed");
		return;
	}

	/* Set the client socket to non-blocking mode. */
    pServer = reinterpret_cast<LwcmServer *>(_pThis);
	if (pServer->SetNonBlocking(client_fd) < 0) {
        PRINT_ERROR("", "ERROR: failed to set client socket to non-blocking");
		close(client_fd);
		return;
	}

    try {
        dcgm_connection_id_t connectionId;
        pConnection = new LwcmServerConnection(pServer->GetConnectionHandler(), pServer, client_fd, client_addr);
        
        if (0 != pServer->GetConnectionHandler()->AddToConnectionTable(pConnection, &connectionId)) {
            PRINT_ERROR("", "ERROR: failed to add connection entry to the connection table");
            DEBUG_STDERR("ERROR: failed to add connection entry to the connection table");
            return;
        }
        // notify the server object about this new connection accept event
        pServer->OnConnectionAdd(connectionId, (LwcmServerConnection*)pConnection);
    } catch (std::runtime_error &e) {
        PRINT_ERROR("%s", "ERROR: %s",  e.what());

        if (pConnection) {
            delete pConnection;
            pConnection = NULL;
        }

        close(client_fd);
        return;
    }
}

/*****************************************************************************
 The LwcmServerConnection class uses this method to add the connection to
 a worker queue
 *****************************************************************************/
void LwcmServer::AddRequestToQueue(LwcmServerConnection* pConnection, void* args)
{
    workqueue_add_job(&mWorkQueue, reinterpret_cast<job_t *>(args));
}

/*****************************************************************************/
LwcmConnectionHandler *LwcmServer::GetConnectionHandler()
{
    return mpConnectionHandler;
}

/*****************************************************************************
 This method is used to set socket as a non-blocking socket
 *****************************************************************************/
int LwcmServer::SetNonBlocking(int fd) {
	int flags;

	flags = fcntl(fd, F_GETFL);
	if (flags < 0) return flags;
	flags |= O_NONBLOCK;
	if (fcntl(fd, F_SETFL, flags) < 0) return -1;
	return 0;
}

/*****************************************************************************
 This method returns the event base maintained by the server
 *****************************************************************************/
struct event_base * LwcmServer::GetEventBase() {
    return mpEvbase;
}

/*****************************************************************************
 * This method is used to read event buffers
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t LwcmServer::ReadEventBuffer(struct bufferevent *bev, void *data, size_t size)
{
    size_t buf_len;
    bufferevent_lock(bev);
    buf_len = bufferevent_read(bev, data, size);
    bufferevent_unlock(bev);
    return buf_len;
}

/*****************************************************************************
 * This method is used to get length of event buffer
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t LwcmServer::GetEventBufferLength(struct bufferevent *bev)
{
    size_t buf_len;

    bufferevent_lock(bev);
    buf_len = evbuffer_get_length(bufferevent_get_input(bev));
    bufferevent_unlock(bev);
    
    return buf_len;
}

/*****************************************************************************/
void LwcmServer::Lock()
{
    lwosEnterCriticalSection(&mMutex);
}

/*****************************************************************************/
void LwcmServer::UnLock()
{
    lwosLeaveCriticalSection(&mMutex);
}

/*****************************************************************************
 * Constructor
 *****************************************************************************/
LwcmServerConnection::LwcmServerConnection(LwcmConnectionHandler *pConnHandler, 
                                                    LwcmServer *pServer, int fd, struct sockaddr_in remoteAddr)
                            : LwcmConnection(pConnHandler)
{
    mFd = fd;
    mpLwcmServerInstance = pServer;
    m_persistAfterDisconnect = false;

    bcopy((char *)&remoteAddr, (char *)&mRemoteSocketAddr, sizeof(mRemoteSocketAddr));

    SetConnectionState(DCGM_CONNECTION_UNKNOWN);

    if ((mpInputBuffer = evbuffer_new()) == NULL) {
        throw std::runtime_error("Error: failed to create input event buffer");
    }

	/* Add any custom code anywhere from here to the end of this function
	 * to initialize your application-specific attributes in the client struct. */
    if ((mpOutputBuffer = evbuffer_new()) == NULL) {
        throw std::runtime_error("Error: failed to create output event buffer");
	}


    // todo: Is it needed to set a timeout if the client is inactive for long?
	// bufferevent_settimeout(mpBufEv, 10, 10);

    mpBufEv = bufferevent_socket_new(pServer->GetEventBase(), mFd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    if (NULL == mpBufEv) {
        throw std::runtime_error("Error: failed to create buffer event");
    }

    SetConnectionState(DCGM_CONNECTION_ACTIVE);

    bufferevent_setcb(mpBufEv, LwcmServer::BufferReadCB, LwcmServer::BufferWriteCB,
                                LwcmServer::BufferEventCB, this);
    bufferevent_enable(mpBufEv, EV_READ|EV_WRITE);

}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
LwcmServerConnection::~LwcmServerConnection() {

    if (mFd > 0) {
        close(mFd);
        mFd = -1;
    }

    if (mpBufEv != NULL) {
        bufferevent_free(mpBufEv);
        mpBufEv = NULL;
    }

    if (mpOutputBuffer != NULL) {
        evbuffer_free(mpOutputBuffer);
        mpOutputBuffer = NULL;
    }

    if (mpInputBuffer != NULL) {
        evbuffer_free(mpInputBuffer);
        mpInputBuffer = NULL;
    }
}

/*****************************************************************************
    Callback registered with libevent. This method is called when data has
    been read from the socket and is available to the application
 *****************************************************************************/
void LwcmServer::BufferReadCB(struct bufferevent *bev, void *_pThis)
{
    unsigned int msgId;
    size_t numBytes;
    int msgType, msgLength, st;
    dcgm_message_header_t lwcmMsgHdr;
    LwcmMessage *pLwcmMessage = NULL;
    LwcmRequest *pLwcmServerRequest = NULL;
    dcgm_request_id_t requestId;

    LwcmServerConnection *pConnection = reinterpret_cast<LwcmServerConnection *>(_pThis);
    if (NULL == pConnection) {
        DEBUG_STDERR("Invalid reference to DCGM connection");
        return;
    }

    size_t buf_len = GetEventBufferLength(bev);
    // cout << "Length of Buffer to read : " << buf_len << endl;

    while (buf_len > 0) {
        switch (pConnection->GetReadState()) {
            case DCGM_CONNECTION_READ_HDR:
                /* Header is not there yet. Return without action*/
                if (buf_len < sizeof (lwcmMsgHdr)) {
                    return;
                }

                /* Read the message header first and get the message type and the
                   size of message to be received */
                numBytes = ReadEventBuffer(bev, &lwcmMsgHdr, sizeof (lwcmMsgHdr));
                if (0 == numBytes) {
                    PRINT_ERROR("", "Failed to get Message Header from the packet");
                    return;
                }

                /* Adjust the Buf length available to be read */
                buf_len = buf_len - numBytes;

                // cout << "Number of bytes received for header read : " << numBytes << "bytes" << endl;
                msgId = ntohl(lwcmMsgHdr.msgId);
                msgType = ntohl(lwcmMsgHdr.msgType);
                requestId = ntohl(lwcmMsgHdr.requestId);
                msgLength = ntohl(lwcmMsgHdr.length);
                if(msgLength < 0 || msgLength > DCGM_PROTO_MAX_MESSAGE_SIZE)
                {
                    PRINT_ERROR("%d", "Got bad message size %d. Closing connection.", msgLength);
                    pConnection->CloseConnection();
                    return;
                }

                if (msgId != DCGM_PROTO_MAGIC) {
                    PRINT_ERROR("%X", "Unexpected DCGM Proto ID x%X", msgId);
                    pConnection->CloseConnection();
                    return;
                }

                /* Allocate a new DCGM Message */
                pLwcmMessage = new LwcmMessage;
                pLwcmMessage->CreateDataBuf(msgLength);
                pLwcmMessage->SetRequestId(requestId);
                pConnection->SetLwrrentHandledMessage(pLwcmMessage);

                pLwcmServerRequest = new LwcmServerRequest(requestId);
                pLwcmServerRequest->ProcessMessage(pLwcmMessage);

                st = pConnection->AddRequest(requestId, pLwcmServerRequest);
                if(st)
                {
                    PRINT_ERROR("%d %u %u", "Got error %d from AddRequest of requestId %u for connectionId %u", 
                                st, requestId, pConnection->GetConnectionId());
                    pConnection->CloseConnection();
                }
                
                pConnection->SetReadState(DCGM_CONNECTION_READ_CONTENT);

                /* Fall through is intentional to get the parts of contents already
                   received */
            case DCGM_CONNECTION_READ_CONTENT: /* Intentional fall-through */

                pLwcmMessage = pConnection->GetLwrrentHandledMessage();

                /* Length of buffer to be read is less than the expected content size
                   then return without any action */
                if (buf_len < pLwcmMessage->GetLength()) {
                    return;
                }

                /* Read Length of message. Make sure the complete message is received */
                /* Read buffer for the specified length */
                numBytes = ReadEventBuffer(bev, pLwcmMessage->GetContent(), pLwcmMessage->GetLength());
                // cout << "Number of bytes received : " << numBytes << "bytes";
                if (numBytes != pLwcmMessage->GetLength()) {
                    PRINT_ERROR("", "Failed to read complete message");
                    DEBUG_STDERR("Failed to read complete message");
                    return;
                }

                buf_len = buf_len - numBytes;
                break;

            default:
                /* This should never happen */
                return;
        }

        /* Change the state machine to represent that msg header is expected next */
        pConnection->SetReadState(DCGM_CONNECTION_READ_HDR);

        /* Push request to the queue */
        job_t *pJob = new job_t;
        LwcmRequestInfo_t *pRequestInfo = new LwcmRequestInfo_t;

        /* Increment the connection's reference counter because we're about to copy a pointer to it */
        pConnection->IncrReference();

        pRequestInfo->requestId = pLwcmMessage->GetRequestId();
        pRequestInfo->pConnection = pConnection;

        pJob->job_function = LwcmServer::ProcessRequest;
        pJob->user_data = pRequestInfo;
        pConnection->GetServer()->AddRequestToQueue(pConnection, pJob);
    }
}

/*****************************************************************************
    This method is called when the write buffer has reached a low watermark.
	That usually means that when the write buffer is 0 length,
	this callback will be called.  It must be defined, but you
	don't actually have to do anything in this callback.
 *****************************************************************************/
void LwcmServer::BufferWriteCB(struct bufferevent *bev, void *_pThis)
{

}

/*****************************************************************************
    This method is called when there is a socket error.  This is used to
    detect that the client disconnected or other socket errors.
 *****************************************************************************/
void LwcmServer::BufferEventCB(struct bufferevent *bev, short events, void *ptr)
{
    LwcmServerConnection *pConnection = reinterpret_cast<LwcmServerConnection *>(ptr);

    if (events & (BEV_EVENT_ERROR|BEV_EVENT_EOF))
    {
        if (pConnection) {
            pConnection->CloseConnection();
            /* Don't use pConnection after this point. It could have been freed */
        }
    }
}

/*****************************************************************************
 This method is ilwoked when the thread is ready to process the request
 *****************************************************************************/
void LwcmServer::ProcessRequest(job_t *pJob)
{
    LwcmRequestInfo_t *pRequestInfo;

    pRequestInfo = (LwcmRequestInfo_t *) pJob->user_data;

    LwcmServerConnection *pConnection = pRequestInfo->pConnection;
    dcgm_request_id_t requestId = pRequestInfo->requestId;

    // DEBUG_STDOUT("Start processing Request");

    /* Ilwoke Worker Function */
    pConnection->GetServer()->OnRequest(requestId, pConnection);

    /* When the control reaches here then it implies one of the following :
     *  1) The blocking request is completed. (Response queued to libevent's output buffer)
     *  2) The async request is addressed by send an ack back.
     *
     * In either of the case, we free up the container holding reference to
     * connection and request-id.
     * For blocking requests, the processing in OnRequest takes care of calling
     * CompleteRequest which removes the request-id entry from the map.
     * For aysnc request, the module handling the request must ilwoke CompleteRequest
     * when it's completed.
     */
    
    /* Decrease the reference count of pConnection since pRequestInfo is going away */
    pConnection->DecrReference();

    delete pRequestInfo;
    delete pJob;
    pRequestInfo = NULL;
    pJob = NULL;
}


/*****************************************************************************/
void LwcmServerConnection::DisableConnectionNotifications()
{
    if (mpBufEv) {
        bufferevent_disable(mpBufEv, EV_READ|EV_WRITE);
    }
}

/*****************************************************************************
 * This method gets the server instance
 *****************************************************************************/
LwcmServer* LwcmServerConnection::GetServer()
{
    return mpLwcmServerInstance;
}

struct sockaddr_in LwcmServerConnection::GetRemoteSocketAddr()
{
    return mRemoteSocketAddr;
}

/*****************************************************************************/
int LwcmServerConnection::SetOutputBuffer(LwcmMessage *pLwcmMsg)
{
    dcgm_message_header_t *pMsgHdr;
    void *buf;
    
    if (!IsConnectionActive()) {
        return -1;
    }
    

    bufferevent_lock(mpBufEv);
    pMsgHdr = pLwcmMsg->GetMessageHdr();
    buf = pLwcmMsg->GetContent();    

    evbuffer_add(mpOutputBuffer, pMsgHdr, sizeof(*pMsgHdr));
    evbuffer_add(mpOutputBuffer, buf, ntohl(pMsgHdr->length));
    
    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        PRINT_ERROR("", "Failed to write message to event buffer");
        DEBUG_STDERR("Failed to write message to event buffer");
        bufferevent_unlock(mpBufEv);
        return -2;
    }
    
    bufferevent_unlock(mpBufEv);
    return 0;    
}

/*****************************************************************************/
void LwcmServerConnection::SetPersistAfterDisconnect(bool value)
{
    m_persistAfterDisconnect = value;
}

/*****************************************************************************/
bool LwcmServerConnection::GetPersistAfterDisconnect(void)
{
    return m_persistAfterDisconnect;
}

/*****************************************************************************/
void LwcmServerConnection::CloseConnection()
{
    SetConnectionState(DCGM_CONNECTION_MARK_TO_CLOSE);
    // notify the server object about this connection close event
    LwcmServer* pServer = GetServer();
    pServer->OnConnectionRemove(GetConnectionId(), this);
    
    //Tell this connection to remove itself from the connection table, which will DecrReference it
    RemoveFromConnectionTable();
}

/*****************************************************************************/
