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
// lwwatch WinDbg Extension
// vpr.h
//
//*****************************************************

#ifndef _LWWATCH_VPR_H_
#define _LWWATCH_VPR_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "rmlsfm.h"
#include "gpuanalyze.h"

void      vprDisplayHelp(void);
LwBool    vprIsSupported(LwU32 indexGpu);
LwBool    vprIsActive(LwU32 indexGpu);
LW_STATUS vprMmuLwrrentRangeInfo(LwU32 indexGpu, LwBool);
LW_STATUS vprBsiMaxRangeInfo(LwU32 indexGpu, LwBool);
LW_STATUS vprBsiLwrrentRangeInfo(LwU32 indexGpu, LwBool);
void      vprPrintMemLockStatus(LwU32 indexGpu, LwBool);
LW_STATUS vprMemLockRangeInfo(LwU32 indexGpu, LwBool);
void      vprPrintBsiType1LockStatus(LwU32 indexGpu, LwBool);
LW_STATUS vprGetAllInfo(LwU32 indexGpu);
LW_STATUS vprGetHwFuseVersions(LwU32 indexGpu);
LW_STATUS vprGetUcodeVersions(LwU32 indexGpu);

#endif // _LWWATCH_VPR_H_
