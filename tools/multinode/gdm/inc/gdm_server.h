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

#include "lwdiagutils.h"
#include "inc/bytestream.h"

namespace GdmServer
{
    LwDiagUtils::EC Start(UINT32 port, UINT32 expectedConnections);
    LwDiagUtils::EC RunOnce();
    UINT32 GetNumConnections();
    LwDiagUtils::EC Shutdown();
    LwDiagUtils::EC SendMessage(ByteStream &bs, void *pvConnection);
    LwDiagUtils::EC BroadcastMessage(ByteStream &bs);
}
