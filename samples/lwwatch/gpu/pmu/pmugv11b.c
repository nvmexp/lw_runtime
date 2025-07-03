/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmugv11b.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"

#include "g_pmu_private.h"     // (rmconfig)  implementation prototypes

const char *
pmuUcodeName_GV11B()
{
    return "g_c85b6_gv11b";
}
