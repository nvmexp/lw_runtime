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

#pragma once

#include "server.h"
#include "transport_common.h"
#include "connection_libevent.h"
#include "event2/event.h"
#include "event2/bufferevent.h"
#include "event2/listener.h"
#include "event2/util.h"

#include <string>
#include <vector>
#include <netinet/in.h>

// A server implementation using libevent
class LibEventServer : public Server
{
public:
    LibEventServer() = default;
    virtual ~LibEventServer() { Shutdown(); }

    LwDiagUtils::EC Initialize
    (
        UINT32              port,
        UINT32              expectedNumClients,
        UINT32              heartbeatMs,
        MessageCallbackFunc msgCallback,
        EventCallbackFunc   eventCallback
    );
    LwDiagUtils::EC Run() override;
    LwDiagUtils::EC RunOnce() override;
    LwDiagUtils::EC Shutdown() override;
    LwDiagUtils::EC SendMessage(const ByteStream & message, Connection *pConnection) override;
    LwDiagUtils::EC BroadcastMessage(const ByteStream & message) override;
    UINT32          GetNumConnections() const override;
private:
    static void ListenerCallback
    (
        struct evconnlistener* pListener
       ,evutil_socket_t        socket
       ,struct sockaddr *      pSaddr
       ,int                    socklen
       ,void *                 pvServer
    );

    static void ReadCallback(struct bufferevent*, void * pvConnection);
    static void EventCallback(struct bufferevent*, short, void * pvConnection);

    vector<LibEventConnection> m_Connections;
    MessageCallbackFunc        m_pMessageCallback = nullptr;
    EventCallbackFunc          m_pEventCallback = nullptr;
    UINT32                     m_ExpectedConnections = 0;
    UINT32                     m_HeartbeatMs = 10;

    struct sockaddr_in      m_Sin;
    struct event_base *     m_pBase        = nullptr;
    struct evconnlistener * m_pListener    = nullptr;
};

void DummyCB(evutil_socket_t fd, short events, void *ptr);