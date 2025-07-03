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

#include "connection.h"
#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "transport_common.h"
#include "event2/bufferevent.h"

#include <vector>

// Wrapper around a libevent connection, shared by both client and server code
class LibEventConnection : public Connection
{
public:
    LibEventConnection() = default;
    explicit LibEventConnection
    (
        struct bufferevent * pBev,
        const string &       remoteIp,
        MessageCallbackFunc  pCb,
        void *               pParent
    ) : m_pBev(pBev), m_RemoteIp(remoteIp), m_pMessageCallback(pCb), m_pParent(pParent) { }
    ~LibEventConnection() { }

    void Disconnect();
    LwDiagUtils::EC SendMessage(const ByteStream & message) override;
    LwDiagUtils::EC HandleInput();
    const string & GetConnectionString() override { return m_RemoteIp; }
    void * GetParent() const { return m_pParent; }

private:
    enum class InputState : UINT08
    {
        WAIT_FOR_HEADER,
        WAIT_FOR_CONTENT
    };
    struct bufferevent  * m_pBev             = nullptr;
    string                m_RemoteIp;
    MessageCallbackFunc   m_pMessageCallback = nullptr;
    void                * m_pParent          = nullptr;
    InputState            m_InputState       = InputState::WAIT_FOR_HEADER;
    ByteStream            m_ReceivedMsg;
};
