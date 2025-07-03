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
// msenc.h
//
//*****************************************************

#ifndef _MSENC_H_
#define _MSENC_H_

#include "os.h"

#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define LWWATCH_MSENC_0             0
#define LWWATCH_MSENC_1             1
#define LWWATCH_MSENC_2             2
#define LWWATCH_MAX_MSENC           3

// method array defines
#define SYSMETHODARRAYSIZE          9
#define CMNMETHODARRAYSIZE         16
#define APPMETHODARRAYSIZE         16

#define CMNMETHODARRAYSIZEC1B7     21
#define APPMETHODARRAYSIZEC1B7     32

#define CMNMETHODARRAYSIZEC4B7     28
#define APPMETHODARRAYSIZEC4B7     32

#define CMNMETHODARRAYSIZEC9B7     32
#define APPMETHODARRAYSIZEC9B7     67

#define GetAppMthdParam(mthd)               ((LW_90B7_MSENC_##mthd - 0x700) >> 2) 
#define GetCmnMthdParam(mthd)               ((LW_90B7_MSENC_##mthd - 0x400) >> 2) 

#define APPMETHODBASE  0x700
#define CMNMETHODBASE  0x400

#define GetAppMthdParam_v02(mthd)           ((LW_A0B7_MSENC_##mthd - 0x400) >> 2) 
#define GetCmnMthdParam_v02(mthd)           ((LW_A0B7_MSENC_##mthd - 0x700) >> 2) 

#define APPMETHODBASE_v02  0x400
#define CMNMETHODBASE_v02  0x700

typedef struct _dbg_msenc_v01_01
{
    LwU32 m_id;
    char *m_tag;
} dbg_msenc_v01_01;

extern dbg_msenc_v01_01 *pMsencMethodTable;
extern dbg_msenc_v01_01 *pMsencFuseReg;
extern dbg_msenc_v01_01 *pMsencPrivReg[LWWATCH_MAX_MSENC];

extern LwU32 lwencId;
extern LwU32 engineId;
extern LwU32 cmnMethodArraySize;
extern LwU32 appMethodArraySize;

#define privInfo_msenc_v01_01(x) {x,#x}

// MSENC non-hal support
BOOL        msencIsSupported(LwU32 indexGpu);
LW_STATUS   msencDumpPriv(LwU32 indexGpu);
LW_STATUS   msencDumpFuse(LwU32 indexGpu);
LW_STATUS   msencDumpImem(LwU32 indexGpu, LwU32 imemSize);
LW_STATUS   msencDumpDmem(LwU32 indexGpu, LwU32 dmemSize);
LW_STATUS   msencTestState(LwU32 indexGpu);
LW_STATUS   msencDisplayHwcfg(LwU32 indexGpu);
void        msencDisplayHelp(void);
LW_STATUS   msencDisplayFlcnSPR(LwU32 indexGpu);
void        msencPrintMethodData_v01_00(LwU32 clmn, char *tag, LwU32 method, LwU32 data);
#endif // _VIC_H_
