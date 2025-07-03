/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch Common OS LwDump
// lwdump_priv.h
//
//*****************************************************

#ifndef _LWDUMP_PRIV_H_
#define _LWDUMP_PRIV_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <lwtypes.h>
#include <lwdump.h>
#include "lwdzip.h"

typedef struct
{
    const char *name;
    LwU32       id;
} LWDUMP_COMPONENT_INFO;

typedef struct
{
    LwU32 gpuIndex;
    const LWDUMP_COMPONENT_INFO *pComponent;
    LwBool bDumpAll;
    LWD_ZIP_HANDLE hZipFile;
    LwU64 configAddr;
    LWDUMP_CONFIG config;
} LWDUMP_STATE;

LW_STATUS lwDumpSetup(LwU32 gpuIndex, const char *pComponentName,
                      const char *pFilename);
LW_STATUS lwDumpCheck();

//
// These are the OS-dependent LwDump routines, and must be supplied
// by OS-specific files. The proper place is the lwdump<OS_NAME>.c(pp)
// file for the OS in question, such as lwdumpWin.cpp.
//
LW_STATUS osLwDumpGetConfigAddr(LwU64* pAddr);
LW_STATUS osLwDumpSetup(LWDUMP_STATE* pState);
LW_STATUS osLwDumpContinue(LWDUMP_STATE* pState);
LW_STATUS osLwDumpReset(LWDUMP_STATE* pState);

#ifdef __cplusplus
}
#endif

#endif // _LWDUMP_PRIV_H_

