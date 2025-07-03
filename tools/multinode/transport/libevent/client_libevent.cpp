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

#include "client_libevent.h"
#include "event2/buffer.h"
#include "event2/thread.h"
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::Initialize
(
    const string &      ipaddr,
    UINT32              port,
    UINT32              heartbeatMs,
    MessageCallbackFunc msgCallback,
    EventCallbackFunc   eventCallback
)
{
    if (m_pBase != nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriLow, "Client already initialized, skipping.\n");
        return LwDiagUtils::OK;
    }

    evthread_use_pthreads();

    m_pBase = event_base_new();
    if (!m_pBase)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Cannot create libevent base for client.\n");
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    memset(&m_Sin, 0, sizeof(m_Sin));
    m_Sin.sin_addr.s_addr = inet_addr(ipaddr.c_str());
    m_Sin.sin_family = AF_INET;
    m_Sin.sin_port   = htons(port);

    m_pMessageCallback = msgCallback;
    m_pEventCallback   = eventCallback;
    m_HeartbeatMs      = heartbeatMs;

    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::Connect()
{
    if (m_pBase == nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Initialize client before attempting to connect.\n");
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    struct bufferevent *pBev = bufferevent_socket_new(m_pBase, -1, BEV_OPT_CLOSE_ON_FREE);
    if (pBev == nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Creating a socket for the client failed\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(m_Sin.sin_addr), ipstr, INET_ADDRSTRLEN);
    m_Connection = LibEventConnection(pBev,
                                      string(ipstr),
                                      m_pMessageCallback,
                                      this);

    //Connect the server
    if (-1 == bufferevent_socket_connect(pBev, (struct sockaddr*)&m_Sin, sizeof(m_Sin)))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to connect to the server\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    if (m_HeartbeatMs != 0)
    {
        // set the client heartbeat
        const struct timeval t = { 0, m_HeartbeatMs * 1000 };
        if (-1 == bufferevent_set_timeouts(pBev, &t, &t))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to set read/write timeouts\n");
            return LwDiagUtils::NETWORK_CANNOT_CONNECT;
        }
    }

    bufferevent_setcb(pBev, ReadCallback, nullptr, EventCallback, &m_Connection);

    if (-1 == bufferevent_enable(pBev, EV_READ | EV_WRITE))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to enable buffer events\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    evutil_socket_t sockFd = bufferevent_getfd(pBev);
    if (sockFd < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to get socket file descriptor\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    int flags = fcntl(sockFd, F_GETFL);
    if (flags < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to get socket flags\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }
    flags |= O_NONBLOCK;
    if (fcntl(sockFd, F_SETFL, flags) < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to set socket flags\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    flags = 1;
    // Allow the socket to send data even when the message is small
    if (setsockopt(sockFd, IPPROTO_TCP, TCP_NODELAY, (char *) &flags, sizeof(int)) < 0)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to set TCP caching options\n");
        return LwDiagUtils::NETWORK_CANNOT_CONNECT;
    }

    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::Run()
{
    if (m_pBase == nullptr)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "Initialize client before attempting to run it.\n");
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    // Call event base dispatch, this function will not return until the client
    // is disconnected
    if (-1 == event_base_dispatch(m_pBase))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Client event loop failed.\n");
        return LwDiagUtils::NETWORK_ERROR;
    }
    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::RunOnce()
{
    // Using MASSERT here instead of a check because this function is designed
    // to be called repetitively
    LWDASSERT(m_pBase != nullptr);

    // Call event base dispatch, this function will not return until the server
    // is shutdown
    if (-1 == event_base_loop(m_pBase, EVLOOP_NONBLOCK))
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Client event loop failed.\n");
        return LwDiagUtils::NETWORK_ERROR;
    }
    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::Disconnect()
{
    m_Connection.Disconnect();
    if (m_pBase != nullptr)
    {
        event_base_free(m_pBase);
        m_pBase = nullptr;
    }

    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventClient::SendMessage(const ByteStream & message)
{
    LwDiagUtils::EC ec;
    ec = m_Connection.SendMessage(message);
    return ec;
}

// -----------------------------------------------------------------------------
void LibEventClient::ReadCallback(struct bufferevent *pBev, void* pvConnection)
{
    LibEventConnection * pConnection = static_cast<LibEventConnection *>(pvConnection);
    LwDiagUtils::EC ec = pConnection->HandleInput();
    if (ec != LwDiagUtils::OK)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError, "Handling input input failed (%d).\n", ec);
    }
}

// -----------------------------------------------------------------------------
void LibEventClient::EventCallback(struct bufferevent * pBev, short events, void* pvConnection)
{
    LibEventConnection * pConnection = static_cast<LibEventConnection *>(pvConnection);
    UINT32               eventTypes = TransportCommon::EventType::NONE;

    if (events & BEV_EVENT_ERROR)
        eventTypes |= TransportCommon::EventType::ERROR;
    if (events & BEV_EVENT_EOF)
        eventTypes |= TransportCommon::EventType::EXITING;
    if (events & BEV_EVENT_CONNECTED)
        eventTypes |= TransportCommon::EventType::CONNECTED;
    if (events & BEV_EVENT_TIMEOUT)
    {
        if (-1 == bufferevent_enable(pBev, EV_READ | EV_WRITE | EV_PERSIST))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "Failed to re-enable buffer events\n");
            eventTypes |= TransportCommon::EventType::ERROR;
        }
        eventTypes |= TransportCommon::EventType::TIMEOUT;
    }

    LibEventClient * pClient = static_cast<LibEventClient *>(pConnection->GetParent());
    if (pClient->m_pEventCallback)
    {
        LwDiagUtils::EC ec = pClient->m_pEventCallback(pConnection, eventTypes);
        if (ec != LwDiagUtils::OK)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "Event callback failed (%d).\n", ec);
        }
    }

    // Libevent disconnects the client when either of these events occur (EOF is
    // a clean exit) so clean up the connection
    if ((eventTypes & TransportCommon::EventType::EXITING) ||
        (eventTypes & TransportCommon::EventType::ERROR))
    {
        pConnection->Disconnect();
    }
}

// -----------------------------------------------------------------------------
GdmClient * GdmClient::CreateLibEventClient
(
    string              ipaddr,
    UINT32              port,
    UINT32              heartbeatMs,
    MessageCallbackFunc msgCallback,
    EventCallbackFunc   eventCallback
)
{
    LibEventClient *pClient = new LibEventClient;
    if (LwDiagUtils::OK != pClient->Initialize(ipaddr, port, heartbeatMs,
                                               msgCallback, eventCallback))
    {
        delete pClient;
        return nullptr;
    }
    return pClient;
}
