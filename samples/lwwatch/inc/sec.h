/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// sec.h
//
//*****************************************************

#ifndef _SEC_H_
#define _SEC_H_

#include "os.h"

#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "g_cipher_private.h"     // (rmconfig)  implementation prototypes

// method array defines
#define SYSMETHODARRAYSIZE   9
#define CMNMETHODARRAYSIZE  16
#define APPMETHODARRAYSIZE  16

#ifdef GetAppMthdParam
#undef GetAppMthdParam
#endif
#define GetAppMthdParam(mthd)               ((LW_95A1_SEC_##mthd - 0x700) >> 2)

#ifdef GetCmnMthdParam
#undef GetCmnMthdParam
#endif
#define GetCmnMthdParam(mthd)               ((LW_95A1_SEC_##mthd - 0x400) >> 2) 

#define APPMETHODBASE  0x700
#define CMNMETHODBASE  0x400

typedef struct _dbg_sec_t114
{
    LwU32 m_id;
    char *m_tag;
} dbg_sec_t114;

extern dbg_sec_t114 *pSecMethodTable;
extern dbg_sec_t114 *pSecPrivReg;

#define privInfo_sec_t114(x) {x,#x}

// SEC non-hal support
BOOL    secIsSupported(LwU32 indexGpu);
LW_STATUS   secDumpPriv(LwU32 indexGpu);
LW_STATUS   secDumpImem(LwU32 indexGpu, LwU32 imemSize);
LW_STATUS   secDumpDmem(LwU32 indexGpu, LwU32 dmemSize);
LW_STATUS   secTestState(LwU32 indexGpu);
LW_STATUS   secDisplayHwcfg(LwU32 indexGpu);
void    secDisplayHelp(void);
LW_STATUS   secDisplayFlcnSPR(LwU32 indexGpu);
void    secPrintMethodData_t114(LwU32 clmn, char *tag, LwU32 method, LwU32 data);
#endif // _SEC_H_
