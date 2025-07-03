/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// acr.h
//
//*****************************************************

#ifndef _LWWATCH_ACR_H_
#define _LWWATCH_ACR_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "rmlsfm.h"
#include "acr_status_codes.h"
#include "gpuanalyze.h"

#define ACR_SUB_WPR_RMASK_L2                    0xC
#define ACR_SUB_WPR_WMASK_L2                    0xC
#define ACR_SUB_WPR_RMASK_L3                    0x8
#define ACR_SUB_WPR_WMASK_L3                    0x8
#define ACR_SUB_WPR_RMASK_ALL_LEVELS_DISABLED   0x0
#define ACR_SUB_WPR_WMASK_ALL_LEVELS_DISABLED   0x0

typedef struct def_lsffalc_prop
{
    char*  name;
    LwBool available;
    LwU32  regBase;
    LwBool bFalconEnabled;
    LwU32  regBaseCfga;
    LwU32  regBaseCfgb;
    LwU32  scratchCodeStart;
    LwU32  scratchDataStart;
    LwU32  regCfgPLM;
    LwU32  size;
} LSFALCPROP, *PLSFALCPROP;

extern LSFALCPROP lsFalc[LSF_FALCON_ID_END];

BOOL      acrIsSupported(LwU32 indexGpu);
LW_STATUS acrLsfStatus(LwU32 indexGpu);
LW_STATUS acrGetRegionInfo(LwU32 indexGpu);
LW_STATUS acrRegionStatus(LwU32 indexGpu);
LW_STATUS acrDmemProtection(LwU32 indexGpu);
LW_STATUS acrImemProtection(LwU32 indexGpu);
void      acrDisplayHelp(void);
LW_STATUS acrGetMultipleWprInfo(LwU32 indexGpu);
LW_STATUS acrVerifyMultipleWprStatus(LwU32 indexGpu);
LW_STATUS acrGetSharedWprStatus(LwU32 indexGpu);

#endif // _LWWATCH_ACR_H_
