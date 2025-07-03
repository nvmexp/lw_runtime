/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "gdm_server.h"
#include "gdm_logger.h"

#include "connection.h"
#include "lwdiagutils.h"
#include "server.h"
#include "inc/bytestream.h"

#include "protobuf/pbwriter.h"
#include "message_handler.h"
#include "message_writer.h"

#include <unistd.h>
#include <memory>

namespace
{
    unique_ptr<Server> m_pServer;

    //------------------------------------------------------------------------------
    LwDiagUtils::EC HandleMessage(Connection * pConnection, const ByteStream & message)
    {
        return MessageHandler::HandleMessages(message, pConnection);
    }

    //------------------------------------------------------------------------------
    LwDiagUtils::EC HandleEvent(Connection * pConnection, UINT32 eventTypes)
    {
        if ((eventTypes & TransportCommon::EventType::EXITING) ||
            (eventTypes & TransportCommon::EventType::ERROR))
        {
            GdmLogger::Printf(LwDiagUtils::PriNormal,
                          "Client %s exited%s\n",
                          pConnection->GetConnectionString().c_str(),
                          (eventTypes & TransportCommon::EventType::ERROR) ? " unexpectedly" : "");
        }
        return LwDiagUtils::OK;
    }
}

//------------------------------------------------------------------------------
LwDiagUtils::EC GdmServer::Start(UINT32 port, UINT32 expectedConnections)
{
    // Create the server and start it running in a thread
    m_pServer.reset(Server::CreateLibEventServer(port,
                                                 expectedConnections,
                                                 10,
                                                 HandleMessage,
                                                 HandleEvent));
    return LwDiagUtils::OK;
}

//------------------------------------------------------------------------------
LwDiagUtils::EC GdmServer::RunOnce()
{
    LWDASSERT(m_pServer);
    return m_pServer->RunOnce();
}

//------------------------------------------------------------------------------
UINT32 GdmServer::GetNumConnections()
{
    LWDASSERT(m_pServer);
    return m_pServer->GetNumConnections();
}

//------------------------------------------------------------------------------
LwDiagUtils::EC GdmServer::Shutdown()
{
    GdmLogger::Printf(LwDiagUtils::PriLow, "Server shutting down, sending SHUDOWN to clients\n");

    ByteStream bs;
    auto sd = MessageWriter::Messages::shutdown(&bs);
    {
        sd
            .header()
                .node_id(0U);
    }
    sd.status(0U);
    sd.Finish();

    LwDiagUtils::EC ec = LwDiagUtils::OK;
    FIRST_EC(m_pServer->BroadcastMessage(bs));

    sleep(1);

    FIRST_EC(m_pServer->Shutdown());

    return ec;
}

//------------------------------------------------------------------------------
LwDiagUtils::EC GdmServer::SendMessage(ByteStream &bs, void *pvConnection)
{
    return m_pServer->SendMessage(bs, static_cast<Connection *>(pvConnection));
}

//------------------------------------------------------------------------------
LwDiagUtils::EC GdmServer::BroadcastMessage(ByteStream &bs)
{
    return m_pServer->BroadcastMessage(bs);
}

