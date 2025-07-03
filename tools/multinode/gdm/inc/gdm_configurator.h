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
#include "GlobalFabricManager.h"

namespace GdmConfig
{
    LwDiagUtils::EC GdmGetGFMConfig(GlobalFmArgs_t *gfmArgs);
}