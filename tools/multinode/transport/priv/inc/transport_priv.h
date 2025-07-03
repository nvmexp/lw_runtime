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

#define TRANSPORT_PROTO_MAGIC 0xD1A6BEEF

struct TransportMsgHeader
{
    UINT32 modsMagic;
    UINT32 length;
};
