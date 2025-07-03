/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "server_libevent.h"
#include "event2/buffer.h"
#include "event2/thread.h"
#include <cstring>
#include <signal.h>
#include <algorithm>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

void DummyCB(evutil_socket_t fd, short events, void *ptr)
{
}

// -----------------------------------------------------------------------------
// Initialize the server.  The server will stop listening for connections once
// the expected number of clients is reached.  Passing 0 for the expected number
// of clients means the server will not stop listening
LwDiagUtils::EC LibEventServer::Initialize
(
    UINT32              port,
    UINT32              expectedNumClients,
    UINT32              heartbeatMs,
    MessageCallbackFunc msgCallback,
    EventCallbackFunc   eventCallback
)
{
    if (m_pBase != nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriLow, "Server already initialized, skipping.\n");
        return LwDiagUtils::OK;
    }

    evthread_use_pthreads();

    m_pBase = event_base_new();
    if (!m_pBase)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Cannot create libevent base for the server.\n");
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    memset(&m_Sin, 0, sizeof(m_Sin));
    m_Sin.sin_family = AF_INET;
    m_Sin.sin_port   = htons(port);

    if (m_HeartbeatMs != 0)
    {
        // Add heartbeat event so that when listening for connections it will not
        // block forever on poll
        const struct timeval t = { 0, m_HeartbeatMs * 1000 };
        struct event * ev1;
        ev1 = event_new(m_pBase, -1, EV_PERSIST, DummyCB, NULL);
        if (-1 == event_add(ev1, &t))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Cannot add heartbeat event to the server.\n");
            return LwDiagUtils::NETWORK_CANNOT_BIND;
        }
    }

    m_pListener = evconnlistener_new_bind(
         m_pBase
        ,LibEventServer::ListenerCallback
        ,(void*)this
        ,LEV_OPT_REUSEABLE|LEV_OPT_CLOSE_ON_FREE
        ,-1
        ,(struct sockaddr*)&m_Sin
        ,sizeof(m_Sin)
    );

    if (!m_pListener)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Cannot create listener for the server.\n");
        return LwDiagUtils::NETWORK_CANNOT_BIND;
    }

    m_pMessageCallback      = msgCallback;
    m_pEventCallback        = eventCallback;
    m_HeartbeatMs           = heartbeatMs;
    m_ExpectedConnections   = expectedNumClients;
    m_Connections.reserve(expectedNumClients);

    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventServer::Run()
{
    if (m_pBase == nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Initialize server before attempting to run it.\n");
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    // Call event base dispatch, this function will not return until the server
    // is shutdown
    if (-1 == event_base_dispatch(m_pBase))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Server event loop failed.\n");
        return LwDiagUtils::NETWORK_ERROR;
    }
    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventServer::RunOnce()
{
    // Using MASSERT here instead of a check because this function is designed
    // to be called repetitively
    LWDASSERT(m_pBase != nullptr);

    // Call event base dispatch, this function will not return until the server
    // is shutdown
    if (-1 == event_base_loop(m_pBase, EVLOOP_NONBLOCK))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Server event loop failed.\n");
        return LwDiagUtils::NETWORK_ERROR;
    }
    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventServer::Shutdown()
{
    for (auto & lwrConnection : m_Connections)
    {
        lwrConnection.Disconnect();
    }

    if (m_pListener != nullptr)
        evconnlistener_free(m_pListener);

    if (m_pBase != nullptr)
        event_base_free(m_pBase);

    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventServer::SendMessage(const ByteStream & message, Connection *pConnection)
{
    return pConnection->SendMessage(message);
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventServer::BroadcastMessage(const ByteStream & message)
{
    LwDiagUtils::EC ec = LwDiagUtils::OK;
    LwDiagUtils::EC lwrEc;
    for (auto & lwrCon : m_Connections)
    {
        lwrEc = lwrCon.SendMessage(message);
        if ((ec == LwDiagUtils::OK) && (LwDiagUtils::OK != lwrEc))
            ec = lwrEc;
    }
    return ec;
}

// -----------------------------------------------------------------------------
UINT32 LibEventServer::GetNumConnections() const
{
    return static_cast<UINT32>(m_Connections.size());
}

namespace
{
    class BevFree
    {
    public:
        explicit BevFree(struct bufferevent * pBev) : m_pBev(pBev) { }
        ~BevFree() { if (m_pBev) bufferevent_free(m_pBev); }
        void Cancel() { m_pBev = nullptr; }
    private:
        struct bufferevent * m_pBev = nullptr;
    };
}
// -----------------------------------------------------------------------------
// This will be called each time a client connects.  The server will stop listening
// for connections once the number of connections specified in initialize is reached
void LibEventServer::ListenerCallback
(
     struct evconnlistener* pListener
    ,evutil_socket_t        socket
    ,struct sockaddr *      pSaddr
    ,int                    socklen
    ,void *                 pvServer
)
{
    LibEventServer     * pServer = static_cast<LibEventServer *>(pvServer);
    struct event_base  * pBase = (struct event_base*) pServer->m_pBase;
    struct bufferevent * pBev;

    LWDASSERT(socklen == sizeof(sockaddr_in));
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET,
              &(reinterpret_cast<struct sockaddr_in *>(pSaddr)->sin_addr),
              ipstr,
              INET_ADDRSTRLEN);

    pBev = bufferevent_socket_new(pBase, socket, BEV_OPT_CLOSE_ON_FREE);
    if (pBev == nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Failed to create a new socket for client connection from %s\n",
                            ipstr);
        return;
    }
    BevFree bevFree(pBev);

    if (-1 == bufferevent_enable(pBev, EV_READ | EV_WRITE | EV_PERSIST))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Failed to set enable buffer events for client connection from %s\n",
                            ipstr);
        return;
    }

    if (pServer->m_HeartbeatMs != 0)
    {
        // set the server heartbeat
        const struct timeval t = { 0, pServer->m_HeartbeatMs * 1000 };
        if (-1 == bufferevent_set_timeouts(pBev, &t, &t))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to set read/write timeouts\n");
            return;
        }
    }

    evutil_socket_t sockFd = bufferevent_getfd(pBev);
    int flags = 1;
    // Allow the socket to send data even when the message is small
    if (setsockopt(sockFd, IPPROTO_TCP, TCP_NODELAY, (char *) &flags, sizeof(int)) < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Failed to set TCP caching options for client connection from %s\n",
                            ipstr);
        return;
    }

    flags = fcntl(sockFd, F_GETFL);
    if (flags < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Failed to get socket flags for client connection from %s\n",
                            ipstr);
        return;
    }
    flags |= O_NONBLOCK;
    if (fcntl(sockFd, F_SETFL, flags) < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Failed to set socket flags for client connection from %s\n",
                            ipstr);
        return;
    }

    pServer->m_Connections.push_back(LibEventConnection(pBev,
                                                        string(ipstr),
                                                        pServer->m_pMessageCallback,
                                                        pServer));
    auto & conn = pServer->m_Connections.back();
    bufferevent_setcb(pBev,
                      LibEventServer::ReadCallback,
                      nullptr,
                      LibEventServer::EventCallback,
                      (void*)&conn);

    bevFree.Cancel();

    LwDiagUtils::Printf(LwDiagUtils::PriLow, "Received client connection from %s\n", ipstr);

    // Stop listening for connections once the expected number of connections is reached
    if (static_cast<UINT32>(pServer->m_Connections.size()) == pServer->m_ExpectedConnections)
    {
        evconnlistener_free(pListener);
        pServer->m_pListener = nullptr;
    }
}

// -----------------------------------------------------------------------------
void LibEventServer::ReadCallback(struct bufferevent *pBev, void* pvConnection)
{
    LibEventConnection * pConnection = static_cast<LibEventConnection *>(pvConnection);
    LwDiagUtils::EC ec = pConnection->HandleInput();
    if (ec != LwDiagUtils::OK)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Handling input input failed (%d).\n", ec);
    }
}

// -----------------------------------------------------------------------------
void LibEventServer::EventCallback(struct bufferevent * pBev, short events, void* pvConnection)
{
    LibEventConnection * pConn   = static_cast<LibEventConnection *>(pvConnection);
    LibEventServer     * pServer = static_cast<LibEventServer *>(pConn->GetParent());
    UINT32               eventTypes = TransportCommon::EventType::NONE;

    if (events & BEV_EVENT_ERROR)
        eventTypes |= TransportCommon::EventType::ERROR;
    if (events & BEV_EVENT_EOF)
        eventTypes |= TransportCommon::EventType::EXITING;
    if (events & BEV_EVENT_CONNECTED)
        eventTypes |= TransportCommon::EventType::CONNECTED;
    if (events & BEV_EVENT_TIMEOUT)
    {
        if (-1 == bufferevent_enable(pBev, EV_READ | EV_WRITE))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to re-enable buffer events\n");
            eventTypes |= TransportCommon::EventType::ERROR;
        }
        eventTypes |= TransportCommon::EventType::TIMEOUT;
    }

    if ((eventTypes & TransportCommon::EventType::EXITING) ||
        (eventTypes & TransportCommon::EventType::ERROR))
    {
        LwDiagUtils::Printf((eventTypes & TransportCommon::ERROR) ? LwDiagUtils::PriError :
                                                                    LwDiagUtils::PriLow,
                            "Client %s exited.\n", pConn->GetConnectionString().c_str());
        auto pRemoveConn = find_if(pServer->m_Connections.begin(),
                                   pServer->m_Connections.end(),
                                   [pConn] (const auto & c) -> bool { return &c == pConn; });
        pRemoveConn->Disconnect();
        pServer->m_Connections.erase(pRemoveConn);
    }

    if (pServer->m_pEventCallback)
    {
        pServer->m_pEventCallback(pConn, eventTypes);
    }
}

// -----------------------------------------------------------------------------
Server * Server::CreateLibEventServer
(
    UINT32 port,
    UINT32 expectedNumClients,
    UINT32 heartbeatMs,
    MessageCallbackFunc msgCallback,
    EventCallbackFunc   eventCallback
)
{
    LibEventServer *pServer = new LibEventServer;
    if (LwDiagUtils::OK != pServer->Initialize(port,
                                               expectedNumClients,
                                               heartbeatMs,
                                               msgCallback,
                                               eventCallback))
    {
        delete pServer;
        return nullptr;
    }
    return pServer;
}
