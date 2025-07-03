/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// hda.h
//
//*****************************************************

#ifndef _HDA_H_
#define _HDA_H_

#include "os.h"

#include "hal.h"
#include "falcon.h"

#include "g_hda_private.h"     // (rmconfig)  implementation prototypes

// HDA non-hal support
BOOL    hdaIsSupported(LwU32 indexGpu);
LW_STATUS   hdaDumpImem(LwU32 indexGpu, LwU32 imemSize);
LW_STATUS   hdaDumpDmem(LwU32 indexGpu, LwU32 dmemSize);
LW_STATUS   hdaTestState(LwU32 indexGpu);
void    hdaDisplayHelp(void);
#endif // _HDA_H_
