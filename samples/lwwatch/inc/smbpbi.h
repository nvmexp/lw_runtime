/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SMBPBI_H_
#define _SMBPBI_H_

#include "os.h"
#include "hal.h"

typedef struct
{
    LwU32 cmd;
    LwU32 dataIn;
    LwU32 dataOut;
} SMBPBI_CONTEXT;

extern LwBool SmbpbiPostedCommandPending;
extern LwU32  SmbpbiMutexToken;

#include "g_smbpbi_hal.h"     // (rmconfig)  public interfaces

void  smbpbiExec             (char *pCmd);
void  smbpbiExecClearContext (void);

LwU32     smbpbiGetCapabilities (LwU8 dwordIdx);
LW_STATUS smbpbiExelwteCommand  (SMBPBI_CONTEXT *pContext);
void      smbpbiReleaseInterface(void);

#endif // _SMBPBI_H_

