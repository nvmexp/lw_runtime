/*
 * File:   FmServerConnection.cpp
 */
#include <sys/types.h>
#ifdef __linux__
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <err.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>

#include <errno.h>
#include <event.h>
#include <thread.h>
#include <signal.h>
#include "workqueue.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "fm_log.h"
#include "FmSocketMessage.h"
#include "FMErrorCodesInternal.h"
#include "FmServerRequest.h"
#include "FmServerConnection.h"
#include "FMCommonTypes.h"

using namespace std;

#if defined(_WINDOWS)
#include <afunix.h>
#endif

ofstream FmSocket::mDebugLogFile;

/*****************************************************************************
 * Constructor
 *****************************************************************************/
FmSocket::FmSocket(int port, char *sockpath, int isTCP, int numWorkers) {
    mPortNumber = port;
    mIsConnectionTCP = isTCP;
    mNumWorkers = numWorkers;

    memset(mSocketPath, 0, sizeof(mSocketPath));
    if(sockpath)
        strncpy(mSocketPath, sockpath, sizeof(mSocketPath)-1);

    mpEvbase = NULL;
    mListenSockFd = -1;
    mServerRunState = FM_SERVER_NOT_STARTED;
	lwosInitializeCriticalSection(&mMutex);
    lwosCondCreate(&mCondition);

#ifdef __linux__
    (void)evthread_use_pthreads();
#else
    evthread_use_windows_threads();
#endif
	if ((mpEvbase = event_base_new()) == NULL) {
        FM_LOG_ERROR("server connection: unable to allocate socket accept event base");
        mListenSockFd = -1;
        throw std::runtime_error("server connection: unable to allocate socket accept event base");
	}    

    mpConnectionHandler = new FmConnectionHandler();
}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
FmSocket::~FmSocket() {
    
    /* Since this is the last object to be deleted by the Host Engine. 
     * Other modules must have removed any references to the pending connection 
     * entries, so it's safe to decrement the count for all the connection entries
     * still present in the list. Ilwoke destructor of connection handler 
     * to decrement reference counts for all the pending connections (if any) */
    if (mpConnectionHandler) {
        delete mpConnectionHandler;    
    }

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
        FM_LOG_WARNING("server connection: killing socket listener thread after stop request timeout");
        Kill();
    }

    if (!mIsConnectionTCP) {
        unlink(mSocketPath);
    }
}

/*****************************************************************************
 * This method is used to record libevent debug/warning events to a file
 * The message is added for debugging purpose.
 *****************************************************************************/
void FmSocket::RecordDebugCB(int severity, const char *msg)
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
void FmSocket::DiscardDebugCB(int severity, const char *msg)
{
    // Do Nothing
}

/*****************************************************************************
 * This method is used to initialize a TCP socket
 *****************************************************************************/
int FmSocket::InitTCPSocket()
{
	struct sockaddr_in listen_addr;
    int reuseaddr_on;

	/* Create our listening socket. */
	mListenSockFd = socket(AF_INET, SOCK_STREAM, 0);
    if (mListenSockFd < 0) {
        FM_LOG_ERROR("server connection: tcp socket object creation failed");
        return -1;
    }

#ifdef _DEBUG
    /* Register to log the warnings/error to a file */
    if (FM_ENABLE_LIBEVENT_LOGGING) {
        char *fileName = (char *)"/tmp/libevent-logs";
        unlink(fileName);
        mDebugLogFile.open(fileName);
        if (!mDebugLogFile.is_open()) {
            FM_LOG_ERROR("ERROR:  Debug file creation failed. Will not log libevent");
            event_set_log_callback(DiscardDebugCB);
        }
        else
            event_set_log_callback(RecordDebugCB);
    } else {
        event_set_log_callback(DiscardDebugCB);
    }
#endif

    reuseaddr_on = 1;
    if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuseaddr_on, sizeof(reuseaddr_on)))
    {
        FM_LOG_ERROR("server connection: failed to set socket property (SO_REUSEADDR). errno %d", errno);
        evutil_closesocket(mListenSockFd);
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
    	    FM_LOG_ERROR("server connection: unable to colwert provided socket path \"%s\" to a network address.", mSocketPath);
	        evutil_closesocket(mListenSockFd);
            mListenSockFd = -1;
            return -1;
	    }
	}

	listen_addr.sin_port = htons(mPortNumber);
	if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
		char *ipStr = inet_ntoa(listen_addr.sin_addr);
        FM_LOG_ERROR("server connection: socket bind operation failed for address %s port %d, with errno %d",
                     ipStr, mPortNumber, errno);
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
	}
    FM_LOG_DEBUG("Port number: %d", mPortNumber);
    return 0;
}

/*****************************************************************************
 * This method is used to initialize a Unix domain socket
 *****************************************************************************/
int FmSocket::InitUnixSocket()
{
	struct sockaddr_un listen_addr;
	int reuseaddr_on;

	/* Create our listening socket. */
	mListenSockFd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (mListenSockFd < 0) {
        FM_LOG_ERROR("server connection: unix domain socket object creation failed");
        return -1;
	}

#ifdef _DEBUG
    /* Register to log the warnings/error to a file */
    if (FM_ENABLE_LIBEVENT_LOGGING) {
        char *fileName = (char *)"/tmp/libevent-logs";
        unlink(fileName);
        mDebugLogFile.open(fileName);
        if (!mDebugLogFile.is_open()) {
            FM_LOG_ERROR("ERROR:  Debug file creation failed. Will not log libevent");
            event_set_log_callback(DiscardDebugCB);
        }
        else
            event_set_log_callback(RecordDebugCB);
    } else {
        event_set_log_callback(DiscardDebugCB);
    }
#endif

	reuseaddr_on = 1;
	if (setsockopt(mListenSockFd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuseaddr_on, sizeof(reuseaddr_on)))
    {
        FM_LOG_ERROR("server connection: failed to set socket property (SO_REUSEADDR). errno %d", errno);
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

	memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sun_family = AF_UNIX;
    strncpy(listen_addr.sun_path, mSocketPath, sizeof(listen_addr.sun_path)-1);
    unlink(mSocketPath); /* Make sure the path doesn't exist or bind will fail */
    if (bind(mListenSockFd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) < 0) {
        FM_LOG_ERROR("server connection: socket bind operation failed for domain socket path %s, errno %d",
                     listen_addr.sun_path, errno);
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }
    
    return 0;
}

/*****************************************************************************
 This method is used to start the listening socket, registers with libevent
 and listens for incoming connections
 *****************************************************************************/
int FmSocket::Serve()
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
        FM_LOG_ERROR("server connection: socket initialization request failed");
        if (mListenSockFd != -1) {
            evutil_closesocket(mListenSockFd);
            mListenSockFd = -1;
        }
        return -1;
    }

	if (listen(mListenSockFd, FM_SERVER_CONNECTION_BACKLOG) < 0) {
        FM_LOG_ERROR("server connection: socket listen request failed");
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
	}

    /* Set the socket to non-blocking, this is essential in event
     * based programming with libevent. */
    if (this->SetNonBlocking(mListenSockFd)) {
        FM_LOG_ERROR("server connection: failed to set server socket non-blocking property");
        evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
        return -1;
    }

	/* Initialize work queue. */
	if (workqueue_init(&mWorkQueue, mNumWorkers)) {
        FM_LOG_ERROR("server connection: failed to create work queue for socket message processing");
		evutil_closesocket(mListenSockFd);
        mListenSockFd = -1;
		return -1;
	}

	/* We now have a listening socket, we create a read event to
	 * be notified when a client connects. */
    event_set(&ev_accept, mListenSockFd, EV_READ|EV_PERSIST, FmSocket::OnAccept, this);
	event_base_set(mpEvbase, &ev_accept);
	event_add(&ev_accept, NULL);

    /* Signal the waiting thread to notify that the server is running to receive connections */
    Lock();
    mServerRunState = FM_SERVER_RUNNING;
    lwosCondSignal(&mCondition);
    UnLock();

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

    /* Signal that server has stopped and will not receive/process any packets */
    Lock();
    mServerRunState = FM_SERVER_STOPPED;
    lwosCondSignal(&mCondition);
    UnLock();

    /* Shut down the work queue last */
    workqueue_shutdown(&mWorkQueue);
	return 0;
}

/*****************************************************************************
 * Implements run method extended from FmThread
 *****************************************************************************/
void FmSocket::run(void)
{
    int ret;
    ret = Serve();
    if (0 != ret) {
        FM_LOG_DEBUG("server connection: server() call failed");
    }
}


/*****************************************************************************/
int FmSocket::WaitToStart()
{
    unsigned int timeout = 1000; /* 1 seconds timeout */
    int ret;

    /* Wait for timeout to get the server started. Return an error if server
       can't be started */
    Lock();
    while (mServerRunState == FM_SERVER_NOT_STARTED) {
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
void FmSocket::StopServer()
{
    unsigned int timeout = 1000; /* 1 seconds timeout */

    /* Wait for server to be stopped */
    Lock();
    if (mServerRunState != FM_SERVER_RUNNING) {
        /* The server instance may not have started or already stopped.
           Don't attempt to break the eventloop in this case as it 
           may get stuck */
        UnLock();
        return;
    }

    /* exit event loop and wait for server thread to stop */
    Stop();
    event_base_loopexit(mpEvbase, NULL); 
    while (mServerRunState != FM_SERVER_STOPPED)
        (void)lwosCondWait(&mCondition, &mMutex, timeout);
    UnLock();
}

/*****************************************************************************
  Callback registered with libevent to accept the incoming connection
 *****************************************************************************/
void FmSocket::OnAccept(evutil_socket_t fd, short ev, void *_pThis)
{
	int client_fd;
	struct sockaddr_in client_addr;
	socklen_t client_len = sizeof(client_addr);
    FmConnection* pConnection = NULL;
    FmSocket* pServer;

	client_fd = accept(fd, (struct sockaddr *)&client_addr, &client_len);
	if (client_fd < 0) {
        FM_LOG_ERROR("server connection: socket accept operation failed");
		return;
	}

	/* Set the client socket to non-blocking mode. */
    pServer = reinterpret_cast<FmSocket *>(_pThis);
	if (pServer->SetNonBlocking(client_fd) < 0) {
        FM_LOG_ERROR("server connection: failed to set client socket non-blocking property");
		evutil_closesocket(client_fd);
		return;
	}

    try {
        fm_connection_id_t connectionId;
        pConnection = new FmServerConnection(pServer->GetConnectionHandler(), pServer, client_fd, client_addr);
        
        if (0 != pServer->GetConnectionHandler()->AddToConnectionTable(pConnection, &connectionId)) {
            FM_LOG_ERROR("server connection: failed to add socket connection entry to the connection tracking table");
            return;
        }
        // notify the server object about this new connection accept event
        pServer->OnConnectionAdd(connectionId, (FmServerConnection*)pConnection);
    } catch (std::runtime_error &e) {
        FM_LOG_ERROR("ERROR: %s",  e.what());

        if (pConnection) {
            delete pConnection;
            pConnection = NULL;
        }

        evutil_closesocket(client_fd);
        return;
    }
}

/*****************************************************************************
 The FmServerConnection class uses this method to add the connection to
 a worker queue
 *****************************************************************************/
void FmSocket::AddRequestToQueue(FmServerConnection* pConnection, void* args)
{
    workqueue_add_job(&mWorkQueue, reinterpret_cast<job_t *>(args));
}

/*****************************************************************************/
FmConnectionHandler *FmSocket::GetConnectionHandler()
{
    return mpConnectionHandler;
}

/*****************************************************************************
 This method is used to set socket as a non-blocking socket
 *****************************************************************************/
int FmSocket::SetNonBlocking(evutil_socket_t fd) {
    evutil_make_socket_nonblocking(fd);
	return 0;
}

/*****************************************************************************
 This method returns the event base maintained by the server
 *****************************************************************************/
struct event_base * FmSocket::GetEventBase() {
    return mpEvbase;
}

/*****************************************************************************
 * This method is used to read event buffers
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t FmSocket::ReadEventBuffer(struct bufferevent *bev, void *data, size_t size)
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
size_t FmSocket::GetEventBufferLength(struct bufferevent *bev)
{
    size_t buf_len;

    bufferevent_lock(bev);
    buf_len = evbuffer_get_length(bufferevent_get_input(bev));
    bufferevent_unlock(bev);
    
    return buf_len;
}

/*****************************************************************************/
void FmSocket::Lock()
{
    lwosEnterCriticalSection(&mMutex);
}

/*****************************************************************************/
void FmSocket::UnLock()
{
    lwosLeaveCriticalSection(&mMutex);
}

/*****************************************************************************
 * Constructor
 *****************************************************************************/
FmServerConnection::FmServerConnection(FmConnectionHandler *pConnHandler, 
                                       FmSocket *pServer, int fd, struct sockaddr_in remoteAddr)
                                       : FmConnection(pConnHandler)
{
    mFd = fd;
    mpFmServerInstance = pServer;
    m_persistAfterDisconnect = false;

    bcopy((char *)&remoteAddr, (char *)&mRemoteSocketAddr, sizeof(mRemoteSocketAddr));

    SetConnectionState(FM_CONNECTION_UNKNOWN);

    if ((mpInputBuffer = evbuffer_new()) == NULL) {
        FM_LOG_ERROR("server connection: failed to allocate socket input event buffer");
        throw std::runtime_error("server connection: failed to allocate socket input event buffer");
    }

	/* Add any custom code anywhere from here to the end of this function
	 * to initialize your application-specific attributes in the client struct. */
    if ((mpOutputBuffer = evbuffer_new()) == NULL) {
        FM_LOG_ERROR("server connection: failed to allocate socket output event buffer");
        throw std::runtime_error("server connection: failed to allocate socket output event buffer");
	}


    // todo: Is it needed to set a timeout if the client is inactive for long?
	// bufferevent_settimeout(mpBufEv, 10, 10);

    mpBufEv = bufferevent_socket_new(pServer->GetEventBase(), mFd, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    if (NULL == mpBufEv) {
        FM_LOG_ERROR("server connection: failed to set buffer event socket listening events");
        throw std::runtime_error("server connection: failed to set buffer event socket listening events");
    }

    SetConnectionState(FM_CONNECTION_ACTIVE);

    bufferevent_setcb(mpBufEv, FmSocket::BufferReadCB, FmSocket::BufferWriteCB,
                                FmSocket::BufferEventCB, this);
    bufferevent_enable(mpBufEv, EV_READ|EV_WRITE);

}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
FmServerConnection::~FmServerConnection() {

    if (mFd > 0) {
        evutil_closesocket(mFd);
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
void FmSocket::BufferReadCB(struct bufferevent *bev, void *_pThis)
{
    unsigned int msgId;
    size_t numBytes;
    int msgType;
    fm_message_header_t fmMsgHdr;
    FmSocketMessage *pFmMessage = NULL;
    FmRequest *pFmServerRequest = NULL;
    fm_request_id_t requestId;

    FmServerConnection *pConnection = reinterpret_cast<FmServerConnection *>(_pThis);
    if (NULL == pConnection) {
        FM_LOG_ERROR("server connection: invalid socket connection object during socket message reading");
        return;
    }

    size_t buf_len = GetEventBufferLength(bev);

    while (buf_len > 0) {
        switch (pConnection->GetReadState()) {
            case FM_CONNECTION_READ_HDR:
                /* Header is not there yet. Return without action*/
                if (buf_len < sizeof (fmMsgHdr)) {
                    return;
                }

                /* Read the message header first and get the message type and the
                   size of message to be received */
                numBytes = ReadEventBuffer(bev, &fmMsgHdr, sizeof (fmMsgHdr));
                if (0 == numBytes) {
                    FM_LOG_ERROR("server connection: failed to get message header from the received packet");
                    return;
                }

                /* Adjust the Buf length available to be read */
                buf_len = buf_len - numBytes;

                msgId = ntohl(fmMsgHdr.msgId);
                msgType = ntohl(fmMsgHdr.msgType);
                requestId = ntohl(fmMsgHdr.requestId);

                if (msgId != FM_PROTO_MAGIC) {
                    FM_LOG_ERROR("server connection: invalid fabric manager message protocol id/signature found on received packet");
                    pConnection->SetConnectionState(FM_CONNECTION_MARK_TO_CLOSE);
                    pConnection->DecrReference();

                    return;
                }

                /* Allocate a new FM Message */
                pFmMessage = new FmSocketMessage;
                pFmMessage->CreateDataBuf(ntohl(fmMsgHdr.length));
                pFmMessage->SetRequestId(requestId);
                pConnection->SetLwrrentHandledMessage(pFmMessage);

                pFmServerRequest = new FmServerRequest(requestId);
                pFmServerRequest->ProcessMessage(pFmMessage);

                pConnection->AddRequest(requestId, pFmServerRequest);
                pConnection->SetReadState(FM_CONNECTION_READ_CONTENT);

                /* Fall through is intentional to get the parts of contents already
                   received */
            case FM_CONNECTION_READ_CONTENT: /* Intentional fall-through */

                pFmMessage = pConnection->GetLwrrentHandledMessage();

                /* Length of buffer to be read is less than the expected content size
                   then return without any action */
                if (buf_len < pFmMessage->GetLength()) {
                    return;
                }

                /* Read Length of message. Make sure the complete message is received */
                /* Read buffer for the specified length */
                numBytes = ReadEventBuffer(bev, pFmMessage->GetContent(), pFmMessage->GetLength());
                if (numBytes != pFmMessage->GetLength()) {
                    FM_LOG_ERROR("server connection: failed to read fabric manager message payload according to header length");
                    return;
                }

                buf_len = buf_len - numBytes;
                break;

            default:
                /* This should never happen */
                return;
        }

        /* Change the state machine to represent that msg header is expected next */
        pConnection->SetReadState(FM_CONNECTION_READ_HDR);

        /* Push request to the queue */
        job_t *pJob = new job_t;
        FmRequestInfo_t *pRequestInfo = new FmRequestInfo_t;

        /* Increment the connection's reference counter because we're about to copy a pointer to it */
        pConnection->IncrReference();

        pRequestInfo->requestId = pFmMessage->GetRequestId();
        pRequestInfo->pConnection = pConnection;

        pJob->job_function = FmSocket::ProcessRequest;
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
void FmSocket::BufferWriteCB(struct bufferevent *bev, void *_pThis)
{

}

/*****************************************************************************
    This method is called when there is a socket error.  This is used to
    detect that the client disconnected or other socket errors.
 *****************************************************************************/
void FmSocket::BufferEventCB(struct bufferevent *bev, short events, void *ptr)
{
    FmServerConnection *pConnection = reinterpret_cast<FmServerConnection *>(ptr);

    if (events & (BEV_EVENT_ERROR|BEV_EVENT_EOF))
    {
        if (pConnection) {
            pConnection->SetConnectionState(FM_CONNECTION_MARK_TO_CLOSE);
            // notify the server object about this connection close event
            FmSocket* pServer = pConnection->GetServer();
            pServer->OnConnectionRemove(pConnection->GetConnectionId(), pConnection);
            
            //Tell this connection to remove itself from the connection table, which will DecrReference it
            pConnection->RemoveFromConnectionTable();
        }
    }
}

/*****************************************************************************
 This method is ilwoked when the thread is ready to process the request
 *****************************************************************************/
void FmSocket::ProcessRequest(job_t *pJob)
{
    FmRequestInfo_t *pRequestInfo;

    pRequestInfo = (FmRequestInfo_t *) pJob->user_data;

    FmServerConnection *pConnection = pRequestInfo->pConnection;
    fm_request_id_t requestId = pRequestInfo->requestId;

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
    if (pConnection)    
        pConnection->DecrReference();

    delete pRequestInfo;
    delete pJob;
    pRequestInfo = NULL;
}


/*****************************************************************************/
void FmServerConnection::DisableConnectionNotifications()
{
    if (mpBufEv) {
        bufferevent_disable(mpBufEv, EV_READ|EV_WRITE);
    }
}

/*****************************************************************************
 * This method gets the server instance
 *****************************************************************************/
FmSocket* FmServerConnection::GetServer()
{
    return mpFmServerInstance;
}

struct sockaddr_in FmServerConnection::GetRemoteSocketAddr()
{
    return mRemoteSocketAddr;
}

/*****************************************************************************/
int FmServerConnection::SetOutputBuffer(FmSocketMessage *pFmMsg)
{
    fm_message_header_t *pMsgHdr;
    void *buf;
    
    if (!IsConnectionActive()) {
        return -1;
    }
    

    bufferevent_lock(mpBufEv);
    pMsgHdr = pFmMsg->GetMessageHdr();
    buf = pFmMsg->GetContent();    

    evbuffer_add(mpOutputBuffer, pMsgHdr, sizeof(*pMsgHdr));
    evbuffer_add(mpOutputBuffer, buf, ntohl(pMsgHdr->length));
    
    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        FM_LOG_ERROR("server connection: failed to write fabric manager socket message to event buffer");
        bufferevent_unlock(mpBufEv);
        return -2;
    }
    
    bufferevent_unlock(mpBufEv);
    return 0;    
}

/*****************************************************************************/
void FmServerConnection::SetPersistAfterDisconnect(bool value)
{
    m_persistAfterDisconnect = value;
}

/*****************************************************************************/
bool FmServerConnection::GetPersistAfterDisconnect(void)
{
    return m_persistAfterDisconnect;
}

/*****************************************************************************/
