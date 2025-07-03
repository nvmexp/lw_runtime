/* 
 * File:   FmClientConnection.cpp
 */

#include <stdio.h>
#include <sys/types.h>
#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <err.h>
#include <signal.h>
#include <netdb.h>
#include <unistd.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#include <sstream>
#include <stdexcept>
#include "fm_log.h"
#include <event.h>
#include <thread.h>
#include <iostream>
#include "FmRequest.h"
#include "FmClientConnection.h"
#include "FmSocketMessage.h"
#include "FMCommonTypes.h"
#include "timelib.h"

/*****************************************************************************
 * Client Listener constructor to create event base
 *****************************************************************************/
FmClientListener::FmClientListener()
{
#ifdef __linux__
    (void)evthread_use_pthreads();
#else
    evthread_use_windows_threads();
#endif
    mpBase = event_base_new();
    if (!mpBase) {
        FM_LOG_ERROR("client connection: failed to open/allocate event base for socket");
        throw std::runtime_error("client connection: failed to open/allocate event base for socket");
    }
}

/*****************************************************************************
 * Client Listener destructor
 *****************************************************************************/
FmClientListener::~FmClientListener()
{
    // DEBUG_STDOUT("In client Listener destructor");
    StopClientListener();
    
    int st = StopAndWait(60000);
    if (st) {
        FM_LOG_WARNING("client connection: killing socket listener thread after stop request timeout.");
        Kill();
    }

    /* Free mpBase after any libevent workers are gone */
    if (mpBase)
    {
        event_base_free(mpBase);
        mpBase = NULL;
    }	
}

/*****************************************************************************
 * Dummy Callback to keep Client listener alive
 *****************************************************************************/
void FmClientListener::DummyCB(evutil_socket_t fd, short events, void *ptr)
{
    /* This callback is a placeholder to keep FM Client Listener active even when
       there are no recv events configured for the client */
}

/*****************************************************************************
 * Run method for listener thread
 *****************************************************************************/
void FmClientListener::run()
{
    struct timeval timeInSec;
    timeInSec.tv_sec = 10;
    timeInSec.tv_usec = 0;
    struct event * ev1;
    ev1 = event_new(mpBase, -1, EV_PERSIST, FmClientListener::DummyCB, NULL);
    event_add(ev1, &timeInSec);
    
#ifdef SIM_BUILD // MODS SIM
    while (!ShouldStop()) {
        event_base_loop(mpBase, EVLOOP_NONBLOCK);
        /* Ensure that we don't deadlock while running in simulation */
        lwosThreadYield();
    }
#else
    event_base_dispatch(mpBase);
#endif
    
    event_free(ev1);
}

/*****************************************************************************/
void FmClientListener::StopClientListener()
{
    /* Close all connections */
    if(!mpBase)
        return;
    
    Stop();
    
	/* Calling loopexit makes dispatch call in run() to break */
    event_base_loopexit(mpBase, NULL);

    /* Note: the worker thread is not guaranteed to be stopped until you call StopAndWait() */
}

/*****************************************************************************
 * This message is used to get event base
 *****************************************************************************/
struct event_base * FmClientListener::GetBase()
{
    return mpBase;
}

/*****************************************************************************
 * This method uses to read event buffers
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t FmClientListener::ReadEventBuffer(struct bufferevent *bev, void *data, size_t size)
{
    size_t buf_len;
    bufferevent_lock(bev);
    buf_len = bufferevent_read(bev, data, size);
    bufferevent_unlock(bev);
    return buf_len;
}

/*****************************************************************************
 * This method uses to get length of event buffer.
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t FmClientListener::GetEventBufferLength(struct bufferevent *bev)
{
    size_t buf_len;

    bufferevent_lock(bev);
    buf_len = evbuffer_get_length(bufferevent_get_input(bev));
    bufferevent_unlock(bev);
    
    return buf_len;
}

/*****************************************************************************
 * This method serves as callback which is ilwoked by Libevent (Listener thread)
 * when there is a message to be read on the socket
 *****************************************************************************/
void FmClientListener::ReadCB(bufferevent* bev, void* ctx)
{
    unsigned int msgId;
    size_t numBytes;
    fm_message_header_t fmMsgHdr;
    FmSocketMessage *pFmMessage = NULL;
    
    FmClientConnection *pConnection = reinterpret_cast<FmClientConnection *>(ctx);

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
                    FM_LOG_ERROR("client connection: failed to get message header from the received packet");
                    return;
                }

                /* Adjust the Buf length available to be read */
                buf_len = buf_len - numBytes;

                msgId = ntohl(fmMsgHdr.msgId);
                if (msgId != FM_PROTO_MAGIC) {
                    FM_LOG_ERROR("client connection: invalid fabric manager message protocol id/signature found on received packet");
                    return;
                }

                /* Allocate a new FM Message */
                pFmMessage = new FmSocketMessage(&fmMsgHdr);
                pFmMessage->CreateDataBuf(ntohl(fmMsgHdr.length));
                pFmMessage->SetRequestId(ntohl(fmMsgHdr.requestId));
                pConnection->SetLwrrentHandledMessage(pFmMessage);
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
                    FM_LOG_ERROR("client connection: failed to read fabric manager message payload according to header length");
                    return;
                }

                /* Adjust the Buf length available to be read */
                buf_len = buf_len - numBytes;

                break;

            default:
                /* This should never happen */
                break;
        }

        /* If the control reaches this point then we are sure that the entire message 
           is received for this particular connection */

        /* Change the state machine to represent that msg header is expected next. 
           This step is needed if there are additional messages expected to be 
           received on this connection */
        pConnection->SetReadState(FM_CONNECTION_READ_HDR);

        /* Get Request Client Handler for Request ID */
        FmRequest *pClientRequest;

        /* Notify Request Handler that the response is received */
        pClientRequest = pConnection->GetRequest(pFmMessage->GetRequestId());
        if (NULL == pClientRequest) {
            // let the derived class decide on what to do with messages received with no request id
            pConnection->ProcessUnSolicitedMessage(pFmMessage);
        } else {
            pClientRequest->ProcessMessage(pFmMessage);
	}
    }
    
    return;
}

/*****************************************************************************
 * To establish connection with Host Engine
 *****************************************************************************/
FmClientConnection::FmClientConnection(FmConnectionHandler* pConnHandler,
                                       FmClientListener* pClientBase,
                                       char* identifier,
                                       int port_number,
                                       bool tryConnection,
                                       bool addressIsUnixSocket,
                                       int connectionTimeoutMs)
    : FmConnection(pConnHandler)
{
    struct hostent *server;
    int ret;
    
    mRequestId = 0;
    bev = NULL;
    mpInputBuffer = 0;
    mpOutputBuffer = 0;
    mAddressIsUnixSocket = addressIsUnixSocket;

    memset(&m_serv_addr, 0, sizeof(m_serv_addr));
    memset(&m_un_addr, 0, sizeof(m_un_addr));

    if(!addressIsUnixSocket)
    {
        /* TCP/IP */
        FM_LOG_DEBUG("Client trying to connect to %s %d", identifier, port_number);
    
        char *buf = (char*) malloc(sizeof(char) * 2048);
        int err;
        struct hostent ent;
        int rc = gethostbyname_r(identifier, &ent, buf, sizeof(char) * 2048, &server, &err);
        if (server == NULL) {
            std::ostringstream ss;
            ss << "client connection: unable to find host information corresponding to ipaddress/FQDN " << identifier << 
                   " port number " << port_number;
            FM_LOG_ERROR("%s", ss.str().c_str());
            free(buf);
            throw std::runtime_error(ss.str());
        }
        
        m_serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr,
            (char *)&m_serv_addr.sin_addr.s_addr,
            server->h_length);
        m_serv_addr.sin_port = htons(port_number);
        free(buf);
    }
    else
    {
        /* Unix socket */
        FM_LOG_DEBUG("Client trying to connect to unix socket %s" , identifier);
        /* sockaddr_un is interchangeable with sockaddr */

        m_un_addr.sun_family = AF_UNIX;
        strncpy(m_un_addr.sun_path, identifier, sizeof(m_un_addr.sun_path)-1);
    }
    
    if (true == tryConnection) {
        bev = bufferevent_socket_new(pClientBase->GetBase(), -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
        if (NULL == bev) {
            std::ostringstream ss;
            ss << "client connection: failed to allocate buffer event socket object";
            FM_LOG_ERROR("%s", ss.str().c_str());
            Cleanup();
            throw std::runtime_error(ss.str());
        }

        bufferevent_setcb(bev, FmClientListener::ReadCB, NULL, FmClientConnection::EventCB, this);
        bufferevent_enable(bev, EV_READ|EV_WRITE);
        SetConnectionState(FM_CONNECTION_PENDING);

        if(mAddressIsUnixSocket)
            ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_un_addr, sizeof(m_un_addr));
        else
            ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_serv_addr, sizeof(m_serv_addr));

        if (0 != ret) {
            std::stringstream ss;
            ss << "client connection: failed to establish socket connection to address " << identifier; 
            FM_LOG_ERROR("%s", ss.str().c_str());
            Cleanup();
            throw std::runtime_error(ss.str());        
        }
        
        /* Dolwmenting the behavior observed for libevent */
        /**
         * If the connection succeeds, FmClientConnection::EventCB gets BEV_EVENT_CONNECTED. Implies
         * connection is good to go.
         * 
         * When the connection fails due to TCP timeout then FmClientConnection::EventCB is notified 
         * with BEV_EVENT_ERROR first followed by an immediate additional event of BEV_EVENT_CONNECTED. 
         * The additional event BEV_EVENT_CONNECTED doesn't imply that the the connection succeeded.
         */

        /* Wait for five secs to see if libevent sends any notification to the connection */
        if (0 != WaitForConnection(connectionTimeoutMs))
        {
            std::stringstream ss;
            ss << "client connection: timeout oclwred while waiting to establish a socket connection to address " << identifier; 
            FM_LOG_ERROR("%s", ss.str().c_str());
            Cleanup();
            throw std::runtime_error(ss.str());
        }

        /* If connection is not marked as active then throw error back to the user */
        if (!this->IsConnectionActive()) {
            std::stringstream ss;
            ss << "client connection: invalid socket state detected for socket connection to address " << identifier; 
            FM_LOG_ERROR("%s", ss.str().c_str());
            Cleanup();
            throw std::runtime_error(ss.str());
        }
    }
    lwosInitializeCriticalSection(&mRequestTableLock);
    mReadState = FM_CONNECTION_READ_HDR;
}

/*****************************************************************************/
FmClientConnection::~FmClientConnection()
{
    FmClientConnection::Cleanup();
    FmConnection::Cleanup();
}

/*****************************************************************************/
void FmClientConnection::Cleanup() 
{
    FM_LOG_DEBUG("FmClientConnection::Cleanup %p, bev %p", this, bev);
    /* Prevent a race to clean up bev */
    Lock();
    if (bev)
    {
        bufferevent_free(bev);
        bev = NULL;
    }
    UnLock();
}

/*****************************************************************************/
void FmClientConnection::SignalConnStateChange(ConnectionState state)
{
    FM_LOG_DEBUG("FmClientConnection::SignalConnStateChange %p, state %u", 
                this, state);
    /* If we're closing the connection, clean up after ourselves */
    if(state >= FM_CONNECTION_MARK_TO_CLOSE)
    {
        Cleanup();
    }
}

/*****************************************************************************/
void FmClientConnection::DisableConnectionNotifications()
{
    if (bev) 
    {
        bufferevent_disable(bev, EV_READ|EV_WRITE);
    }
}

/*****************************************************************************/
int FmClientConnection::WaitForConnection(unsigned int milliseconds)
{
    int st;
    int ret = 0;
    ConnectionState connState;

    timelib64_t startWait    = timelib_usecSince1970();
    timelib64_t microseconds = (timelib64_t) milliseconds * 1000;

    while ((connState = GetConnectionState()) == FM_CONNECTION_PENDING)
    {
        if (startWait + microseconds < timelib_usecSince1970())
        {
            FM_LOG_DEBUG("Failed to connect: Timed out after %d milliseconds", milliseconds);
            return -1;
        }

        lwosSleep(1); /* Sleep for a ms */
    }

    if (connState == FM_CONNECTION_ACTIVE)
    {
        FM_LOG_DEBUG("WaitForConnection() connected!");
        return 0;
    }
    else
    {
        FM_LOG_DEBUG("Failed to connect: state = %d", (int)connState);
        return -1;
    }
}

/*****************************************************************************/
void FmClientConnection::DisableLibeventCallbacks(void)
{
    if(!bev)
        FM_LOG_DEBUG("DisableLibeventCallbacks: bev was NULL");
    else
    {
        bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
        FM_LOG_DEBUG("DisableLibeventCallbacks: callbacks were disabled");
    }
}

/*****************************************************************************
 * This method serves as a callback to receive events for the connection
 ****************************************************************************/
void FmClientConnection::EventCB(bufferevent* bev, short events, void *ptr) 
{
    FmClientConnection *pConnection = (FmClientConnection *)ptr;

    FM_LOG_DEBUG("In Event CB events x%X, ptr %p", events, ptr);
    
    if (events & BEV_EVENT_CONNECTED) 
    {
        /* This event doesn't mean that we connected successfully. It actually means that we
           completed the process of connecting, whether it was successful or not. Only set the
           connection as active if it is still pending */
        FM_LOG_DEBUG("Connection complete. Event: %x\n", events);
        if(pConnection->GetConnectionState() == FM_CONNECTION_PENDING)
        {
            pConnection->SetConnectionState(FM_CONNECTION_ACTIVE);
        }
    } 
    else if ((events & BEV_EVENT_ERROR) || (events & BEV_EVENT_EOF)) 
    {
        FM_LOG_DEBUG("Connection Error. Event: %x Connection state %d", events, pConnection->GetConnectionState());
        pConnection->SetConnectionState(FM_CONNECTION_MARK_TO_CLOSE);
        pConnection->SetAllRequestsStatus(FM_INT_ST_CONNECTION_NOT_VALID);
        /* We don't want any further callbacks on this connection if it's going away. Otherwise,
           libevent will queue a BEV_EVENT_CONNECTED event right after this that we don't want. 
           Worse, the caller may turn around and destroy bev at the same time, which can deadlock 
           libevent */
        pConnection->DisableLibeventCallbacks();
        pConnection->RemoveFromConnectionTable();
    }
}

/*****************************************************************************/
int FmClientConnection::SetOutputBuffer(FmSocketMessage* pFmMsg)
{
    fm_message_header_t *pMsgHdr;
    void *buf;

    if (!IsConnectionActive()) {
        return -1;
    }
    
    pMsgHdr = pFmMsg->GetMessageHdr();
    buf = pFmMsg->GetContent();
    
    bufferevent_lock(bev);
    evbuffer_add(bufferevent_get_output(bev), pMsgHdr, sizeof(*pMsgHdr));
    evbuffer_add(bufferevent_get_output(bev), buf, ntohl(pMsgHdr->length));
    bufferevent_unlock(bev);
    return 0;
}
