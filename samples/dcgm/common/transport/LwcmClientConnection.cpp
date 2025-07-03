/* 
 * File:   LwcmClientConnection.cpp
 */
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <err.h>
#include <signal.h>
#include <netdb.h>
#include <sys/un.h>

#include <sstream>
#include <stdexcept>
#include "logging.h"
#include <event.h>
#include <thread.h>
#include <iostream>
#include "LwcmRequest.h"
#include "LwcmClientConnection.h"
#include "LwcmProtocol.h"
#include "LwcmSettings.h"
#include "timelib.h"


/*****************************************************************************
 * Client Listener constructor to create event base
 *****************************************************************************/
LwcmClientListener::LwcmClientListener()
{
    (void)evthread_use_pthreads();    
    mpBase = event_base_new();
    if (!mpBase) {
        throw std::runtime_error("Failed to open event base");
    }
}

/*****************************************************************************
 * Client Listener destructor
 *****************************************************************************/
LwcmClientListener::~LwcmClientListener()
{
    // DEBUG_STDOUT("In client Listener destructor");
    StopClientListener();
    
    int st = StopAndWait(60000);
    if (st) {
        PRINT_WARNING("", "Killing Client Listener thread that is still running.");
        DEBUG_STDERR("Killing Client Listener thread that is still running.");
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
void LwcmClientListener::DummyCB(evutil_socket_t fd, short events, void *ptr)
{
    /* This callback is a placeholder to keep DCGM Client Listener active even when
       there are no recv events configured for the client */
}

/*****************************************************************************
 * Run method for listener thread
 *****************************************************************************/
void LwcmClientListener::run()
{
    struct timeval timeInSec;
    timeInSec.tv_sec = 10;
    timeInSec.tv_usec = 0;
    struct event * ev1;
    ev1 = event_new(mpBase, -1, EV_PERSIST, LwcmClientListener::DummyCB, NULL);
    event_add(ev1, &timeInSec);
    
    event_base_dispatch(mpBase);
    
    event_free(ev1);
}

/*****************************************************************************/
void LwcmClientListener::StopClientListener()
{
    /* Close all connections */
    if(!mpBase)
        return;
    
	/* Calling loopexit makes dispatch call in run() to break */
    event_base_loopexit(mpBase, NULL);    

    /* Note: the worker thread is not guaranteed to be stopped until you call StopAndWait() */
}

/*****************************************************************************
 * This message is used to get event base
 *****************************************************************************/
struct event_base * LwcmClientListener::GetBase()
{
    return mpBase;
}

/*****************************************************************************
 * This method uses to read event buffers
 * This method provides thread safe way of using libevent bufferevents.
 *****************************************************************************/
size_t LwcmClientListener::ReadEventBuffer(struct bufferevent *bev, void *data, size_t size)
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
size_t LwcmClientListener::GetEventBufferLength(struct bufferevent *bev)
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
void LwcmClientListener::ReadCB(bufferevent* bev, void* ctx)
{
    unsigned int msgId;
    size_t numBytes;
    dcgm_message_header_t lwcmMsgHdr;
    LwcmMessage *pLwcmMessage = NULL;
    
    LwcmClientConnection *pConnection = reinterpret_cast<LwcmClientConnection *>(ctx);

    size_t buf_len = GetEventBufferLength(bev);

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
                    cout << "Failed to get Message ID from the packet" << endl;
                    return;
                }

                /* Adjust the Buf length available to be read */
                buf_len = buf_len - numBytes;

                msgId = ntohl(lwcmMsgHdr.msgId);
                if (msgId != DCGM_PROTO_MAGIC) {
                    PRINT_ERROR("", "Failed to match DCGM Proto ID");
                    return;
                }

                /* Allocate a new DCGM Message */
                pLwcmMessage = new LwcmMessage(&lwcmMsgHdr);
                pLwcmMessage->CreateDataBuf(ntohl(lwcmMsgHdr.length));
                pLwcmMessage->SetRequestId(ntohl(lwcmMsgHdr.requestId));
                pConnection->SetLwrrentHandledMessage(pLwcmMessage);
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
                if (numBytes != pLwcmMessage->GetLength()) {
                    PRINT_ERROR("", "Failed to read complete message");
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
        pConnection->SetReadState(DCGM_CONNECTION_READ_HDR);

        /* Get Request Client Handler for Request ID */
        LwcmRequest *pClientRequest;

        /* Notify Request Handler that the response is received */
        pClientRequest = pConnection->GetRequest(pLwcmMessage->GetRequestId());
        if (NULL == pClientRequest) {
            // let the derived class decide on what to do with messages received with no request id
            pConnection->ProcessUnSolicitedMessage(pLwcmMessage);
        } else {
            pClientRequest->ProcessMessage(pLwcmMessage);
	}
    }
    
    return;
}

/*****************************************************************************
 * To establish connection with Host Engine
 *****************************************************************************/
LwcmClientConnection::LwcmClientConnection(LwcmConnectionHandler* pConnHandler,
                                           LwcmClientListener* pClientBase,
                                           char* identifier,
                                           int port_number,
                                           bool tryConnection,
                                           bool addressIsUnixSocket,
                                           int connectionTimeoutMs)
    : LwcmConnection(pConnHandler)
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
        PRINT_DEBUG("%s %d", "Client trying to connect to %s %d", identifier, port_number);
    
        server = gethostbyname(identifier);
        if (server == NULL) {
            PRINT_ERROR("%s %d", "Error: No host found corresponding to IPaddress/FQDN %s Port Number %d", identifier, port_number);
            std::stringstream ss;
            ss << "Error: No host found corresponding to " << identifier;
            throw std::runtime_error(ss.str());
        }
        
        m_serv_addr.sin_family = AF_INET;

        bcopy((char *)server->h_addr,
            (char *)&m_serv_addr.sin_addr.s_addr,
            server->h_length);
        m_serv_addr.sin_port = htons(port_number);
    }
    else
    {
        /* Unix socket */
        PRINT_DEBUG("%s", "Client trying to connect to unix socket %s" , identifier);
        /* sockaddr_un is interchangeable with sockaddr */

        m_un_addr.sun_family = AF_UNIX;
        strncpy(m_un_addr.sun_path, identifier, sizeof(m_un_addr.sun_path)-1);
    }
    
    if (true == tryConnection) {
        bev = bufferevent_socket_new(pClientBase->GetBase(), -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
        if (NULL == bev) {
            std::stringstream ss;
            ss << "Failed to create socket";
            Cleanup();
            throw std::runtime_error(ss.str());
        }

        bufferevent_setcb(bev, LwcmClientListener::ReadCB, NULL, LwcmClientConnection::EventCB, this);
        bufferevent_enable(bev, EV_READ|EV_WRITE);
        SetConnectionState(DCGM_CONNECTION_PENDING);
        
        if(mAddressIsUnixSocket)
            ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_un_addr, sizeof(m_un_addr));
        else
            ret = bufferevent_socket_connect(bev, (struct sockaddr *)&m_serv_addr, sizeof(m_serv_addr));

        if (0 != ret) {
            std::stringstream ss;
            ss << "Failed to connect to Host engine running at IP " << identifier; 
            Cleanup();
            throw std::runtime_error(ss.str());        
        }
        
        /* Dolwmenting the behavior observed for libevent */
        /**
         * If the connection succeeds, LwcmClientConnection::EventCB gets BEV_EVENT_CONNECTED. Implies
         * connection is good to go.
         * 
         * When the connection fails due to TCP timeout then LwcmClientConnection::EventCB is notified 
         * with BEV_EVENT_ERROR first followed by an immediate additional event of BEV_EVENT_CONNECTED. 
         * The additional event BEV_EVENT_CONNECTED doesn't imply that the the connection succeeded.
         */

        /* Wait for five secs to see if libevent sends any notification to the connection */
        if (0 != WaitForConnection(connectionTimeoutMs))
        {
            PRINT_DEBUG("", "Connection timeout");
            PRINT_ERROR(
                "%s %d", "Error: Failed to connect to IPaddress/FQDN %s Port Number %d", identifier, port_number);
            std::stringstream ss;
            ss << "Failed to connect to Host engine running at IP " << identifier;
            Cleanup();
            throw std::runtime_error(ss.str());
        }

        /* If connection is not marked as active then throw error back to the user */
        if (!this->IsConnectionActive()) {
            PRINT_DEBUG("", "Can't connect");
            PRINT_ERROR("%s %d", "Error: Failed to connect to IPaddress/FQDN %s Port Number %d", identifier, port_number);
            std::stringstream ss;
            ss << "Failed to connect to Host engine running at IP " << identifier; 
            Cleanup();
            throw std::runtime_error(ss.str());                             
        }
    }
    lwosInitializeCriticalSection(&mRequestTableLock);
    mReadState = DCGM_CONNECTION_READ_HDR;
}

/*****************************************************************************/
LwcmClientConnection::~LwcmClientConnection()
{
    LwcmClientConnection::Cleanup();
    LwcmConnection::Cleanup();
}

/*****************************************************************************/
void LwcmClientConnection::Cleanup() 
{
    PRINT_DEBUG("%p %p", "LwcmClientConnection::Cleanup %p, bev %p", this, bev);
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
void LwcmClientConnection::SignalConnStateChange(ConnectionState state)
{
    PRINT_DEBUG("%p %u", "LwcmClientConnection::SignalConnStateChange %p, state %u", 
                this, state);
    /* If we're closing the connection, clean up after ourselves */
    if(state >= DCGM_CONNECTION_MARK_TO_CLOSE)
    {
        Cleanup();
    }
}

/*****************************************************************************/
void LwcmClientConnection::DisableConnectionNotifications()
{
    if (bev) 
    {
        bufferevent_disable(bev, EV_READ|EV_WRITE);
    }
}

/*****************************************************************************/
int LwcmClientConnection::WaitForConnection(unsigned int milliseconds)
{
    int st;
    int ret = 0;
    ConnectionState connState;
    timelib64_t startWait    = timelib_usecSince1970();
    timelib64_t microseconds = milliseconds * 1000;

    while ((connState = GetConnectionState()) == DCGM_CONNECTION_PENDING)
    {
        if (startWait + microseconds < timelib_usecSince1970())
        {
            PRINT_DEBUG("%d", "Failed to connect: Timed out after %d milliseconds", milliseconds);
            return -1;
        }

        usleep(1000); /* Sleep for a ms */
    }

    if (connState == DCGM_CONNECTION_ACTIVE)
    {
        PRINT_DEBUG("", "WaitForConnection() connected!");
        return 0;
    }
    else
    {
        PRINT_DEBUG("%d", "Failed to connect: state = %d", (int)connState);
        return -1;
    }
}

/*****************************************************************************/
void LwcmClientConnection::DisableLibeventCallbacks(void)
{
    if(!bev)
        PRINT_DEBUG("", "DisableLibeventCallbacks: bev was NULL");
    else
    {
        bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
        PRINT_DEBUG("", "DisableLibeventCallbacks: callbacks were disabled");
    }
}

/*****************************************************************************
 * This method serves as a callback to receive events for the connection
 ****************************************************************************/
void LwcmClientConnection::EventCB(bufferevent* bev, short events, void *ptr) 
{
    LwcmClientConnection *pConnection = (LwcmClientConnection *)ptr;

    PRINT_DEBUG("%X %p", "In Event CB events x%X, ptr %p", events, ptr);
    
    if (events & BEV_EVENT_CONNECTED) 
    {
        /* This event doesn't mean that we connected successfully. It actually means that we
           completed the process of connecting, whether it was successful or not. Only set the
           connection as active if it is still pending */
        PRINT_DEBUG("%x", "Connection complete. Event: %x\n", events);
        if(pConnection->GetConnectionState() == DCGM_CONNECTION_PENDING)
        {
            pConnection->SetConnectionState(DCGM_CONNECTION_ACTIVE);
        }
    } 
    else if ((events & BEV_EVENT_ERROR) || (events & BEV_EVENT_EOF)) 
    {
        PRINT_DEBUG("%x","Connection Error. Event: %x\n", events);
        pConnection->SetConnectionState(DCGM_CONNECTION_MARK_TO_CLOSE);
        pConnection->SetAllRequestsStatus(DCGM_ST_CONNECTION_NOT_VALID);
        /* We don't want any further callbacks on this connection if it's going away. Otherwise,
           libevent will queue a BEV_EVENT_CONNECTED event right after this that we don't want. 
           Worse, the caller may turn around and destroy bev at the same time, which can deadlock 
           libevent */
        pConnection->DisableLibeventCallbacks();
        pConnection->RemoveFromConnectionTable();
    }
}

/*****************************************************************************/
int LwcmClientConnection::SetOutputBuffer(LwcmMessage* pLwcmMsg)
{
    dcgm_message_header_t *pMsgHdr;
    void *buf;

    if (!IsConnectionActive()) {
        return -1;
    }
    
    pMsgHdr = pLwcmMsg->GetMessageHdr();
    buf = pLwcmMsg->GetContent();
    
    bufferevent_lock(bev);
    evbuffer_add(bufferevent_get_output(bev), pMsgHdr, sizeof(*pMsgHdr));
    evbuffer_add(bufferevent_get_output(bev), buf, ntohl(pMsgHdr->length));
    bufferevent_unlock(bev);
    return 0;
}
