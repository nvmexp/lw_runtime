/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef INCLUDED_TEE_H
#define INCLUDED_TEE_H

void Printf(int, const char *Format, ...);

namespace Tee
{
    enum Priority
    {
        PriNone      = 0,
        PriDebug     = 1,
        PriLow       = 2,
        PriNormal    = 3,
        PriHigh      = 4,
        PriAlways    = 5,
        ScreenOnly   = 6,
        FileOnly     = 7,
        SerialOnly   = 8,
        CirlwlarOnly = 9,
        DebuggerOnly = 10,
        EthernetOnly = 11,
        StdoutOnly   = 12
    };

    enum Level
    {
        LevAlways  = 0,
        LevDebug   = 1,
        LevLow     = 2,
        LevNormal  = 3,
        LevHigh    = 4,
        LevNone    = 5,
    };
}

#endif
