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
#include "connection.h"

namespace TransportCommon
{
   enum EventType
   {
       NONE       = 0
      ,EXITING    = (1 << 0)
      ,ERROR      = (1 << 1)
      ,TIMEOUT    = (1 << 2)
      ,CONNECTED  = (1 << 3)
   };
}

typedef LwDiagUtils::EC (*MessageCallbackFunc)(Connection *, const ByteStream &);
typedef LwDiagUtils::EC (*EventCallbackFunc)(Connection *, UINT32);

