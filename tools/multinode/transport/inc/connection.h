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

#pragma once

#include <string>
#include "lwdiagutils.h"
#include "inc/bytestream.h"

// Virtual class describing a connection used in a client/server application
class Connection
{
public:
    virtual ~Connection() { }
    virtual LwDiagUtils::EC SendMessage(const ByteStream & message) = 0;

    // Get a string describing the connection
    virtual const string & GetConnectionString() = 0;
};
