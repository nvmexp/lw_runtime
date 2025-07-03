/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 11.11.2003
// fb.h
//
//*****************************************************

#ifndef _FB_H_
#define _FB_H_

#include "os.h"

//
// Routines that differ per chip
//

LW_STATUS    readSystem(LwU64 pa, void* buffer, LwU32 length);
LW_STATUS    writeSystem(LwU64 pa, void* buffer, LwU32 length);

//
// Used by fbMonitorAccess_GK104
//
typedef struct
{
     char* name;
     LwU32 addr;
     LwU32 value;
     LwU32 mask;
} setup_writes_t;

#include "g_fb_hal.h"                    // (rmconfig) public interface

extern FB_LWHAL_IFACES pFb[MAX_GPUS];

#endif // _FB_H_
