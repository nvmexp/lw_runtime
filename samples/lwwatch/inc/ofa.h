/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/
#ifndef _OFA_H_
#define _OFA_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "g_ofa_hal.h"         // (rmconfig)  public interfaces
#include "g_ofa_private.h"     // (rmconfig)  implementation prototypes

// OFA registers RD/WR in GPU space
#define LWWATCH_OFA_0            0
#define LWWATCH_MAX_OFA          3
#define privInfo_ofa_v01_00(x)   {x,#x}

// method array defines
#define METHODARRAYSIZEC6FA      72
#define METHODARRAYSIZEC9FA      176
#define MAXMETHODARRAYSIZE       176
#define METHODBASE  0x700

#define ILWALID_OFFSET              0xFFFFFFFF

#define UCODE_MAP_FNAME_FORMAT      "g_ofa_%x_%04d.map"
#define VAR_NAME_OFF2MTHDOFFS       "pAppIntData"

typedef struct _dbg_ofa_v01_00
{
    LwU32 m_id;
    char *m_tag;
} dbg_ofa_v01_00;

extern dbg_ofa_v01_00 *pOfaPrivReg[LWWATCH_MAX_OFA];
extern dbg_ofa_v01_00 *pOfaFuseReg;
extern dbg_ofa_v01_00 *pOfaMethodTable;
extern LwU32 ofaId;

//
// Common Non-HAL Functions
//
POBJFLCN           ofaGetFalconObject      (void);
LwU32              ofaGetDmemAccessPort    (void);
const char*        ofaGetEngineName_v01_00 (void);
const char*        ofaGetSymFilePath       (void);

// OFA non-hal support
BOOL        ofaIsGpuSupported(LwU32 indexGpu);
LW_STATUS   ofaDumpPriv(LwU32 indexGpu);
LW_STATUS   ofaDumpFuse(LwU32 indexGpu);
LW_STATUS   ofaDumpImem(LwU32 indexGpu, LwU32 imemSize);
LW_STATUS   ofaDumpDmem(LwU32 indexGpu, LwU32 dmemSize, LwU32 offs2MthdOffs);
LW_STATUS   ofaTestState(LwU32 indexGpu);
LW_STATUS   ofaDisplayHwcfg(LwU32 indexGpu);
void        ofaDisplayHelp(void);
LW_STATUS   ofaDisplayFlcnSPR(LwU32 indexGpu);
LW_STATUS   ofaGetOff2MthdOffs(LwU32 classNum, LwU32 *off2MthdOffs);

#endif // _OFA_H_
