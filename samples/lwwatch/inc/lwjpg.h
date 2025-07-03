/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for LWJPG
// lwjpg.h
//
//*****************************************************

#ifndef _LWWATCH_LWJPG_H_
#define _LWWATCH_LWJPG_H_

#include "os.h"

#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "g_lwjpg_private.h"     // (rmconfig)  implementation prototypes

// from GH100 onwards we have multiple LWJPG engines supported
#define LWWATCH_LWJPG_0             0
#define LWWATCH_LWJPG_1             1
#define LWWATCH_LWJPG_2             2
#define LWWATCH_LWJPG_3             3
#define LWWATCH_LWJPG_4             4
#define LWWATCH_LWJPG_5             5
#define LWWATCH_LWJPG_6             6
#define LWWATCH_LWJPG_7             7
#define LWWATCH_MAX_LWJPG           8

// method array defines
#define SYSMETHODARRAYSIZE_LWJPG  9
#define CMNMETHODARRAYSIZE_LWJPG  42
#define CMNMETHODARRAYSIZE_C4D1   42
#define CMNMETHODBASE_LWJPG_v02   0x700
#define APP_ID_ADDRESS_IN_DMEM    0x154     // 0x134+0x20 (as defined in msdecos.c,
                                            // 0x134 is starting offset of sysMethodArray and
                                            // 0x20 because last element of sysMethodArray contains appMethodID)

typedef struct _dbg_lwjpg_v02_00
{
    LwU32 m_id;
    char *m_tag;
} dbg_lwjpg_v02_00;

extern dbg_lwjpg_v02_00 *pLwjpgMethodTable;
extern dbg_lwjpg_v02_00 *pLwjpgFuseReg;
extern dbg_lwjpg_v02_00 *pLwjpgPrivReg[LWWATCH_MAX_LWJPG];

#define privInfo_lwjpg_v02_00(x) {x,#x}

// LWJPG non-hal support
BOOL        lwjpgIsSupported(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwjpgDumpPriv(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwjpgDumpFuse(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwjpgDumpImem(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize);
LW_STATUS   lwjpgDumpDmem(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize);
LW_STATUS   lwjpgTestState(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwjpgDisplayHwcfg(LwU32 indexGpu, LwU32 engineId);
void        lwjpgDisplayHelp(void);
LW_STATUS   lwjpgDisplayFlcnSPR(LwU32 indexGpu, LwU32 engineId);
void        lwjpgPrintMethodData_v01_00(LwU32 clmn, char *tag, LwU32 method, LwU32 data);
#endif // _LWWATCH_LWJPG_H_
