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

#include "client.h"
#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "transport_common.h"
#include "connection_libevent.h"
#include "event2/event.h"
#include "event2/bufferevent.h"

#include <string>
#include <vector>
#include <netinet/in.h>

// A client implementation using libevent
class LibEventClient : public GdmClient
{
public:
    LibEventClient() = default;
    virtual ~LibEventClient() { Disconnect(); }

    LwDiagUtils::EC Initialize
    (
        const string &      ipaddr,
        UINT32              port,
        UINT32              heartbeatMs,
        MessageCallbackFunc msgCallback,
        EventCallbackFunc   eventCallback
    );
    LwDiagUtils::EC Connect() override;
    LwDiagUtils::EC Run() override;
    LwDiagUtils::EC RunOnce() override;
    LwDiagUtils::EC Disconnect() override;
    LwDiagUtils::EC SendMessage(const ByteStream & message) override;
private:
    static void ReadCallback(struct bufferevent*, void * pvClient);
    static void EventCallback(struct bufferevent*, short, void * pvClient);

    struct sockaddr_in      m_Sin;
    struct event_base *     m_pBase = nullptr;

    MessageCallbackFunc     m_pMessageCallback = nullptr;
    EventCallbackFunc       m_pEventCallback   = nullptr;
    UINT32                  m_HeartbeatMs      = 10;
    LibEventConnection      m_Connection;
};
