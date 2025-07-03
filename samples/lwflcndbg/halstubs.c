/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// amanjunatha@lwpu.com - 7.21.2008
// halstubs.c
//
//*****************************************************

//
// If any of the headers have typedefined structures
// which are being used as parameters to hal functions
// please include those before including g_hal_stubs.h
//
#include "os.h"
#include "hal.h"
#include "pmu.h"
#include "socbrdg.h"
#include "tegrasys.h"
#include "g_hal_stubs.h"

