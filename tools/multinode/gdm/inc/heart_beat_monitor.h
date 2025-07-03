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

namespace HeartBeatMonitor
{
    using MonitorHandle = INT64;
    LwDiagUtils::EC InitMonitor();
    LwDiagUtils::EC SendUpdate(MonitorHandle regId);
    MonitorHandle RegisterApp(UINT32 appId, UINT32 nodeId, UINT64 heartBeatPeriodSeconds);
    LwDiagUtils::EC UnRegisterApp(MonitorHandle regId);
    LwDiagUtils::EC RegisterGdm(void *pvConnection);
    LwDiagUtils::EC SendGdmHb();
};
