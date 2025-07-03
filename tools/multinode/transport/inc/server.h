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

#pragma once

#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "transport_common.h"

// Virtual class describing the interface for a server in a client/server application
class Server
{
public:
    // Run the server, typically this function would be called from a dedicated
    // thread and listen for messages from clients or other events.  It would
    // execute a callback function when necessary to notify the application on
    // reception of a message/event.  Depending on the underlying implementation
    // this function may not return until the server is shutdown
    virtual LwDiagUtils::EC Run() = 0;

    // This method will run the client exactly once checking for things that need
    // processing and then processing them.  This call will not block as Run does
    virtual LwDiagUtils::EC RunOnce()    = 0;

    // Shutdown the server
    virtual LwDiagUtils::EC Shutdown() = 0;

    // Send a message to a specific client
    virtual LwDiagUtils::EC SendMessage(const ByteStream & message, Connection *pConnection) = 0;

    // Broadcast a message to all clients
    virtual LwDiagUtils::EC BroadcastMessage(const ByteStream & message) = 0;

    // Get the total number of clients
    virtual UINT32          GetNumConnections() const = 0;

    // Specific server type creation routines
    static Server * CreateLibEventServer
    (
        UINT32              port,
        UINT32              expectedNumClients,
        UINT32              heartbeatMs,
        MessageCallbackFunc msgCallback,
        EventCallbackFunc   eventCallback
    );
};
