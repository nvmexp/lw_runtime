/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for LWDEC
// lwdec.h
//
//*****************************************************

#ifndef _LWWATCH_LWDEC_H_
#define _LWWATCH_LWDEC_H_

#include "os.h"

#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes

// from Tu10x onwards we have multiple lwdec engines supported
#define LWWATCH_LWDEC_0             0
#define LWWATCH_LWDEC_1             1
#define LWWATCH_LWDEC_2             2
#define LWWATCH_LWDEC_3             3
#define LWWATCH_LWDEC_4             4
#define LWWATCH_LWDEC_5             5
#define LWWATCH_LWDEC_6             6
#define LWWATCH_LWDEC_7             7
#define LWWATCH_MAX_LWDEC           8

// method array defines
#define SYSMETHODARRAYSIZE         9
#define CMNMETHODARRAYSIZE        16
#define APPMETHODARRAYSIZE        16
#define CMNMETHODARRAYSIZEB0B0    27
#define APPMETHODARRAYSIZEB0B0    32
#define CMNMETHODARRAYSIZEB6B0    28
#define APPMETHODARRAYSIZEB6B0    16
#define CMNMETHODARRAYSIZEC1B0    49
#define APPMETHODARRAYSIZEC1B0    10
#define CMNMETHODARRAYSIZEC4B0    50
#define APPMETHODARRAYSIZEC4B0    11

#define APP_ID_ADDRESS_IN_DMEM            0x154     // 0x134+0x20 (as defined in msdecos.c,
                                                    // 0x134 is starting offset of sysMethodArray and
                                                    // 0x20 because last element of sysMethodArray contains appMethodID)

#define APPMETHODBASE_LWDEC_v01           0x400

#define APPMETHODBASE_LWDEC_v02_H264      0x400
#define APPMETHODBASE_LWDEC_v02_VP8       0x480
#define APPMETHODBASE_LWDEC_v02_VC1       0x500
#define APPMETHODBASE_LWDEC_v02_VP9       0x580
#define APPMETHODBASE_LWDEC_v02_MPEG12    0x600
#define APPMETHODBASE_LWDEC_v02_HEVC      0x680
#define APPMETHODBASE_LWDEC_v02_MPEG4     0xE00

#define APPMETHODBASE_LWDEC_v03_H264      0x500
#define APPMETHODBASE_LWDEC_v03_VP8       0x540
#define APPMETHODBASE_LWDEC_v03_HEVC      0x580
#define APPMETHODBASE_LWDEC_v03_VP9       0x5C0
#define APPMETHODBASE_LWDEC_v03_AES_PASS1 0x600
#define APPMETHODBASE_LWDEC_v03_CTR64     0xC00

#define CMNMETHODBASE_LWDEC_v01           0x700
#define CMNMETHODBASE_LWDEC_v03_CODECS    0x400
#define CMNMETHODBASE_LWDEC_v03_CTR64     0xF00

typedef struct _dbg_lwdec_v01_01
{
    LwU32 m_id;
    char *m_tag;
} dbg_lwdec_v01_01;

extern dbg_lwdec_v01_01 *pLwdecMethodTable;
extern dbg_lwdec_v01_01 *pLwdecFuseReg;
extern dbg_lwdec_v01_01 *pLwdecPrivReg[LWWATCH_MAX_LWDEC];

#define privInfo_lwdec_v01_01(x) {x,#x}

// LWDEC non-hal support
BOOL        lwdecIsSupported(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwdecDumpPriv(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwdecDumpFuse(LwU32 indexGpu);
LW_STATUS   lwdecDumpImem(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize);
LW_STATUS   lwdecDumpDmem(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize);
LW_STATUS   lwdecTestState(LwU32 indexGpu, LwU32 engineId);
LW_STATUS   lwdecDisplayHwcfg(LwU32 indexGpu, LwU32 engineId);
void        lwdecDisplayHelp(void);
LW_STATUS   lwdecDisplayFlcnSPR(LwU32 indexGpu, LwU32 engineId);
void        lwdecPrintMethodData_v01_00(LwU32 clmn, char *tag, LwU32 method, LwU32 data);
#endif // _LWWATCH_LWDEC_H_
