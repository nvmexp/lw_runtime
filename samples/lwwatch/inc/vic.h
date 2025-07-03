/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// vic.h
//
//*****************************************************

#ifndef _VIC_H_
#define _VIC_H_

#include <string.h>
#include "hal.h"
#include "tegrasys.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"
#include "os.h"

typedef struct _dbg_vic
{
    LwU32 m_id;
    char *m_tag;
} dbg_vic;

extern dbg_vic *pVicMethodTable;
extern dbg_vic *pVicPrivReg;

#define privInfo_vic(x) {x,#x}

#define MAX_SLOTS 5
// method array defines
#define SYSMETHODARRAYSIZE   9
#define CMNMETHODARRAYSIZE  16
#define APPMETHODARRAYSIZE  16

#define APPMETHODBASE  0x700
#define CMNMETHODBASE  0x400

// VIC non-hal support
BOOL    vicIsSupported(LwU32 indexGpu);
LW_STATUS   vicDumpPriv(LwU32 indexGpu);
LW_STATUS   vicDumpImem(LwU32 indexGpu);
LW_STATUS   vicDumpDmem(LwU32 indexGpu);
LW_STATUS   vicTestState(LwU32 indexGpu);
LW_STATUS   vicDisplayHwcfg(LwU32 indexGpu);
void    vicDisplayHelp(void);
LW_STATUS   vicDisplayFlcnSPR(LwU32 indexGpu);
void    vicPrintMethodData_t124(LwU32 clmn, char *tag, LwU32 method, LwU32 data);
#endif // _VIC_H_
