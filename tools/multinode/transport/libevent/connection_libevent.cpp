/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "connection_libevent.h"
#include "event2/buffer.h"
#include "event2/event.h"
#include "transport_priv.h"
#include "transport_common.h"
#include <unistd.h>

// -----------------------------------------------------------------------------
void LibEventConnection::Disconnect()
{
    if (m_pBev)
        bufferevent_free(m_pBev);
    m_pBev = nullptr;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventConnection::SendMessage(const ByteStream & message)
{
    if (m_pBev == nullptr)
        return LwDiagUtils::UNSUPPORTED_FUNCTION;

    bufferevent_lock(m_pBev);
    TransportMsgHeader msgHeader = { TRANSPORT_PROTO_MAGIC, static_cast<UINT32>(message.size()) };

    struct evbuffer * pBuffer = bufferevent_get_output(m_pBev);

    evbuffer_add(pBuffer, &msgHeader, sizeof(msgHeader));
    evbuffer_add(pBuffer, &message[0], message.size());
    bufferevent_flush(m_pBev, EV_WRITE, BEV_NORMAL);
    bufferevent_unlock(m_pBev);
    return LwDiagUtils::OK;
}

// -----------------------------------------------------------------------------
LwDiagUtils::EC LibEventConnection::HandleInput()
{
    if (m_pBev == nullptr)
        return LwDiagUtils::UNSUPPORTED_FUNCTION;

    size_t bufLen;
    bufferevent_lock(m_pBev);
    bufLen = evbuffer_get_length(bufferevent_get_input(m_pBev));
    bufferevent_unlock(m_pBev);

    while (bufLen > 0)
    {
        switch (m_InputState)
        {
            case InputState::WAIT_FOR_HEADER:
                {
                    // Header is not there yet. Return without action
                    if (bufLen < sizeof(TransportMsgHeader))
                        return LwDiagUtils::OK;

                    // Read the message header first and get the message type and the
                    // size of message to be received
                    TransportMsgHeader msgHeader;
                    bufferevent_lock(m_pBev);
                    UINT32 bytesRead = bufferevent_read(m_pBev, &msgHeader, sizeof(msgHeader));
                    bufferevent_unlock(m_pBev);

                    if (sizeof(msgHeader) != bytesRead)
                    {
                        LwDiagUtils::Printf(LwDiagUtils::PriError,
                                            "Failed to get message header from socket\n");
                        return LwDiagUtils::NETWORK_READ_ERROR;
                    }

                    // Adjust the Buf length available to be read
                    bufLen = bufLen - bytesRead;
                    if (msgHeader.modsMagic != TRANSPORT_PROTO_MAGIC)
                    {
                        LwDiagUtils::Printf(LwDiagUtils::PriError,
                                            "Invalid message header signature received\n");
                        return LwDiagUtils::NETWORK_READ_ERROR;
                    }

                    m_ReceivedMsg.resize(msgHeader.length);
                    m_InputState = InputState::WAIT_FOR_CONTENT;
                }

                // fall-through

            case InputState::WAIT_FOR_CONTENT:
                {
                    // Length of buffer to be read is less than the expected content size
                    // then return without any action
                    if (bufLen < m_ReceivedMsg.size())
                    {
                        return LwDiagUtils::OK;
                    }

                    // Read Length of message. Make sure the complete message is received
                    // Read buffer for the specified length
                    bufferevent_lock(m_pBev);
                    UINT32 bytesRead = bufferevent_read(m_pBev,
                                                        &m_ReceivedMsg[0],
                                                        m_ReceivedMsg.size());
                    bufferevent_unlock(m_pBev);

                    if (bytesRead != m_ReceivedMsg.size())
                    {
                        LwDiagUtils::Printf(LwDiagUtils::PriError,
                                            "Failed to read message payload from socket\n");
                        return LwDiagUtils::NETWORK_READ_ERROR;
                    }

                    // Adjust the Buf length available to be read
                    bufLen = bufLen - bytesRead;
                }
                break;

            default:
                // This should never happen
                break;
        }

        // If the control reaches this point then we are sure that the entire message
        // is received for this particular connection

        // Change the state machine to represent that msg header is expected next.
        // This step is needed if there are additional messages expected to be
        // received on this connection
        m_InputState = InputState::WAIT_FOR_HEADER;
        if (m_pMessageCallback)
        {
            LwDiagUtils::EC ec;
            CHECK_EC(m_pMessageCallback(this, m_ReceivedMsg));
        }
    }

    return LwDiagUtils::OK;
}
