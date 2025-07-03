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
#include <set>

namespace GlobalFmManager
{
    LwDiagUtils::EC InitGFM();
    LwDiagUtils::EC ShutDownGFM();

    LwDiagUtils::EC GetNumGpus(void *pvConnection);
    LwDiagUtils::EC GetNumLwSwitch(void *pvConnection);

    LwDiagUtils::EC GetGpuMaxLwLinks(UINT32 physicalId, void *pvConnection);

    LwDiagUtils::EC GetGpuPhysicalId(UINT32 index, void *pvConnection);
    LwDiagUtils::EC GetLwSwitchPhysicalId(UINT32 index, void *pvConnection);

    LwDiagUtils::EC GetGpuEnumIndex(UINT32 nodeID, UINT32 physicalId, void *pvConnection);
    LwDiagUtils::EC GetLwSwitchEnumIndex(UINT32 nodeID, UINT32 physicalId, void *pvConnection);

    LwDiagUtils::EC GetGpuPciBdf(UINT32 nodeID, UINT32 enumIndex, void *pvConnection);
    LwDiagUtils::EC GetLwSwitchPciBdf(UINT32 nodeID, UINT32 enumIndex, void *pvConnection);
};
