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

#include "server.h"
#include "connection.h"
#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "protobuf/pbreader.h"
#include "protobuf/pbwriter.h"
#include "message_reader.h"
#include "message_writer.h"
#include "message_structs.h"
#include "message_handler.h"
#include <memory>
#include <thread>
#include <unistd.h>
#include <termios.h>

// This is a simple client example application, the exelwtable takes 2 parameters:
// the port number to use and the expected number of clients.  It will then wait
// to receive a WAITING message from each client and once one is received send
// the client a GO message, once the client is waiting for DISCONNECT hitting 'q'
// on the command line will tell all clients to disconnect.

//----------------------------- Helper Functions for Input ---------------------
class EnterSingleCharMode
{
public:
    EnterSingleCharMode(bool echo)
    {
        // Disable canonical mode and echo
        struct termios newSettings;
        tcgetattr(fileno(stdin), &m_SavedStdinSettings);
        newSettings = m_SavedStdinSettings;
        newSettings.c_lflag &= ~ICANON;
        if (!echo)
        {
            newSettings.c_lflag &= ~ECHO;
        }
        tcsetattr(fileno(stdin), TCSANOW, &newSettings);
    }
    ~EnterSingleCharMode()
    {
        tcsetattr(fileno(stdin), TCSANOW, &m_SavedStdinSettings);
    }

private:
    struct termios m_SavedStdinSettings = {};
};

//------------------------------------------------------------------------------
namespace
{
    unique_ptr<Server> m_pServer;
    UINT32 m_ExpectedConnections = 0;
    UINT32 m_LwrrentlyWaiting    = 0;
    bool m_bReceivedDisconnect   = false;

    bool KeyboardHit()
    {
        int ready = 0;
        timeval tv;
        fd_set files;
        FD_ZERO(&files);
        // Stream 0 is stdin
        FD_SET(0, &files);

        tv.tv_sec = 0;
        tv.tv_usec = 0;

        ready = select(1, &files, NULL, NULL, &tv);

        return ready ? true : false;
    }

    LwDiagUtils::EC BroadcastFlowControl(MessageWriter::FlowControl::State msg)
    {
        ByteStream bs;
        auto go = MessageWriter::Messages::flow_control(&bs);
        {
            go
                .header()
                    .message_type(MessageWriter::MessageHeader::mt_flow_control)
                    .source_id(0LL)
                    .print("Hello from server");
        }
        go
            .state(msg)
            .error_id(0LL);
        go.Finish();
        sleep(1);
        LwDiagUtils::EC ec = m_pServer->BroadcastMessage(bs);
        m_LwrrentlyWaiting = 0;
        return ec;
    }

    LwDiagUtils::EC HandleMessage(Connection * pConnection, const ByteStream & message)
    {
        return MessageHandler::HandleMessages(message, pConnection);
    }

    LwDiagUtils::EC HandleEvent(Connection * pConnection, UINT32 eventTypes)
    {
        if ((eventTypes & TransportCommon::EventType::EXITING) ||
            (eventTypes & TransportCommon::EventType::ERROR))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                          "Client %s exited%s\n",
                          pConnection->GetConnectionString().c_str(),
                          (eventTypes & TransportCommon::EventType::ERROR) ? " unexpectedly" : "");

            m_ExpectedConnections = m_pServer->GetNumConnections();
            if (m_pServer->GetNumConnections() == 0)
                m_bReceivedDisconnect = true;
        }
        return LwDiagUtils::OK;
    }

    LwDiagUtils::EC ServerThread(Server *pServer)
    {
        return pServer->Run();
    }

    LwDiagUtils::EC ShutdownServer(thread & serverThread)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Received 'q' from keyboard or server shutting down, "
                            "sending DISCONNECT to clients\n");
        LwDiagUtils::EC ec = BroadcastFlowControl(MessageWriter::FlowControl::state_disconnect);
        sleep(1);
        m_pServer->Shutdown();
        if (serverThread.joinable())
            serverThread.join();
        return ec;
    }
}

//------------------------------------------------------------------------------
LwDiagUtils::EC MessageHandler::HandleFlowControl
(
    Messages::FlowControl const &msg,
    void *                       pvConnection
)
{
    switch (msg.header.message_type)
    {
        case MessageReader::MessageHeader::mt_flow_control:
            switch (msg.state)
            {
                case MessageReader::FlowControl::state_waiting:
                    m_LwrrentlyWaiting++;
                    LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                                        "Received WAITING message from client %llu\n",
                                        msg.header.source_id);
                    if (m_LwrrentlyWaiting == m_ExpectedConnections)
                    {
                        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                                            "All clients waiting, broadcasting GO\n");
                        BroadcastFlowControl(MessageWriter::FlowControl::state_go);
                    }
                    break;

                case MessageReader::FlowControl::state_error:
                case MessageReader::FlowControl::state_go:
                    break;
                case MessageReader::FlowControl::state_disconnect:
                    m_bReceivedDisconnect = true;
                    break;
            }
            break;
        default:
            break;
    }
    if (!msg.header.print.empty())
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Message from client %llu : %s\n",
                            msg.header.source_id,
                            msg.header.print.c_str());
    }
    return LwDiagUtils::OK;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Usage : %s port expected_connections\n",
                            argv[0]);
        return LwDiagUtils::BAD_PARAMETER;
    }

    // Create the server and start it running in a thread
    m_ExpectedConnections = static_cast<UINT32>(atoi(argv[2]));
    m_pServer.reset(Server::CreateLibEventServer(atoi(argv[1]),
                                                 m_ExpectedConnections,
                                                 10,
                                                 HandleMessage,
                                                 HandleEvent));

    thread t(ServerThread, m_pServer.get());

    // Wait until all clients have connected
    struct timespec ts = { 0, 100000000 };
    while (m_pServer->GetNumConnections() != m_ExpectedConnections)
    {
        nanosleep(&ts, NULL);
    }

    EnterSingleCharMode e(false);
    for (;;)
    {
        while (KeyboardHit())
        {
            char c = getchar();
            if ((c == 'q') || m_bReceivedDisconnect)
            {
                return ShutdownServer(t);
            }

            if (c == 'g')
            {
                LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                                    "Received 'g' broadcasting out of band GO to clients\n");
                BroadcastFlowControl(MessageWriter::FlowControl::state_go);
            }
        }
        if (m_bReceivedDisconnect)
        {
            return ShutdownServer(t);
        }
    }

    if (t.joinable())
        t.join();

    return 0;
}
