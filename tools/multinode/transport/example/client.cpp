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

#include "client.h"
#include "connection.h"
#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "protobuf/pbreader.h"
#include "protobuf/pbwriter.h"
#include "message_structs.h"
#include "message_handler.h"
#include "message_writer.h"
#include "message_reader.h"
#include <memory>
#include <unistd.h>
#include <thread>
#include <termios.h>

// This is a simple client example application, the exelwtable takes 3 parameters:
// the server IP, port number, and connection ID.  It then waits for user input
// before starting to send waiting messages (to allow the user to start multiple
// clients).  The server application will then send go messages to all clients
// once it has received a waiting message from all connected clients.  After 10
// waiting messages the client will wait for a disconnect message

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
    unique_ptr<GdmClient> m_pClient;
    bool   m_bGoReceived         = false;
    bool   m_bDisconnectReceived = false;
    UINT32 m_ConnectionId        = 0;

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

    LwDiagUtils::EC HandleMessage(Connection * pConnection, const ByteStream & message)
    {
        return MessageHandler::HandleMessages(message, pConnection);
    }

    LwDiagUtils::EC ClientThread(GdmClient *pClient)
    {
        return pClient->Run();
    }

    LwDiagUtils::EC HandleEvent(Connection * pConnection, UINT32 eventTypes)
    {
        if ((eventTypes & TransportCommon::EventType::EXITING) ||
            (eventTypes & TransportCommon::EventType::ERROR))
        {
            LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                                "Server exited prematurely, exiting client\n");
            m_pClient->Disconnect();
            m_bDisconnectReceived = true;
        }
        return LwDiagUtils::OK;
    }
}

//------------------------------------------------------------------------------
LwDiagUtils::EC MessageHandler::HandleFlowControl
(
    Messages::FlowControl const &msg,
    void * pvConnection
)
{
    switch (msg.header.message_type)
    {
        case MessageReader::MessageHeader::mt_flow_control:
            switch (msg.state)
            {
                case MessageReader::FlowControl::state_waiting:
                case MessageReader::FlowControl::state_error:
                    break;
                case MessageReader::FlowControl::state_go:
                    m_bGoReceived = true;
                    LwDiagUtils::Printf(LwDiagUtils::PriNormal, "GO received\n");
                    break;
                case MessageReader::FlowControl::state_disconnect:
                    LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Disconnect received\n");
                    m_pClient->Disconnect();
                    m_bDisconnectReceived = true;
                    break;
            }
            break;
        default:
            break;
    }

    if (!msg.header.print.empty())
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Message from server %llu : %s\n",
                            msg.header.source_id,
                            msg.header.print.c_str());
    }
    return LwDiagUtils::OK;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Usage : %s ipaddr port connection_id\n",
                            argv[0]);
        return LwDiagUtils::BAD_PARAMETER;
    }

    // Create the client and start it running in a thread
    m_ConnectionId = static_cast<UINT32>(atoi(argv[3]));
    m_pClient.reset(GdmClient::CreateLibEventClient(argv[1],
                                                 atoi(argv[2]),
                                                 10,
                                                 HandleMessage,
                                                 HandleEvent));

    LwDiagUtils::EC ec;
    CHECK_EC(m_pClient->Connect());
    thread t(ClientThread, m_pClient.get());

    EnterSingleCharMode e(false);

    LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                        "Pausing for user input before sending messages\n");
    struct timespec ts = { 0, 100000000 };
    while (!KeyboardHit())
    {
        nanosleep(&ts, NULL);
    }

    for (UINT32 i = 0; i < 10; i++)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Sending WAITING message\n");
        m_bGoReceived = false;

        ByteStream bs;
        auto waiting = MessageWriter::Messages::flow_control(&bs);
        {
            waiting
                .header()
                    .message_type(MessageWriter::MessageHeader::mt_flow_control)
                    .source_id(m_ConnectionId)
                    .print("Hello from client");
        }
        waiting
            .state(MessageWriter::FlowControl::state_waiting)
            .error_id(0LL);
        waiting.Finish();
        CHECK_EC(m_pClient->SendMessage(bs));
        LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Waiting for GO from server\n");

        while (!m_bGoReceived)
        {
            nanosleep(&ts, NULL);
            if (m_bDisconnectReceived)
            {
                LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                                    "Server disconnected early exiting\n");
                if (t.joinable())
                    t.join();
                return LwDiagUtils::NETWORK_ERROR;
            }
        }
        LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Sleeping between messages\n");
        sleep(1);
    }
    LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Waiting for DISCONNECT\n");
    while (!m_bDisconnectReceived)
    {
        nanosleep(&ts, NULL);
    }

    LwDiagUtils::Printf(LwDiagUtils::PriNormal, "DISCONNECTING\n");

    if (t.joinable())
        t.join();
    return 0;
}
