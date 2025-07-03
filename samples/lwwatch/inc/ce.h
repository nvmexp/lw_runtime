/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// ce.h
//
//*****************************************************

#ifndef _CE_H_
#define _CE_H_

#include "os.h"

// CE non-hal support

BOOL    ceIsValid(LwU32 indexGpu, LwU32 indexCe);
BOOL    ceIsSupported(LwU32 indexGpu, LwU32 indexCe);
LW_STATUS   ceDumpPriv(LwU32 indexGpu, LwU32 indexCe);
LW_STATUS   ceTestState(LwU32 indexGpu, LwU32 indexCe);
void    cePrintPceLceMap(LwU32 indexGpu);
void    cePrintPriv(LwU32 clmn, char *tag, LwU32 id);
void    ceDisplayHelp(void);

#include "g_ce_hal.h"     // (rmconfig)  public interfaces

#endif // _CE_H_
