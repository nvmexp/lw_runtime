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

#include "lwdiagutils.h"
#include "inc/bytestream.h"
#include "transport_common.h"
#include <vector>

// Virtual class describing the interface for a client in a client/server application
class GdmClient
{
public:
    // Connect to the server
    virtual LwDiagUtils::EC Connect()    = 0;

    // Run the client, typically this function would be called from a dedicated
    // thread and listen for messages from the server or other events.  It would
    // execute a callback function when necessary to notify the application on
    // reception of such an event.  Depending on the underlying implementation
    // this function may not return until the client is disconnected
    virtual LwDiagUtils::EC Run()        = 0;

    // This method will run the client exactly once checking for things that need
    // processing and then processing them.  This call will not block as Run does
    virtual LwDiagUtils::EC RunOnce()    = 0;

    // Disconnect the client
    virtual LwDiagUtils::EC Disconnect() = 0;

    // Send a message to the server
    virtual LwDiagUtils::EC SendMessage(const ByteStream & message) = 0;

    // Specific client type creation routines
    static GdmClient * CreateLibEventClient
    (
        string              ipaddr,
        UINT32              port,
        UINT32              heartbeatMs,
        MessageCallbackFunc msgCallback,
        EventCallbackFunc   eventCallback
    );
};

