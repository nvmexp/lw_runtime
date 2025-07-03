/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmuga102.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"

#include "g_pmu_private.h"     // (rmconfig)  implementation prototypes

const char *
pmuUcodeName_GA102()
{
    return "g_c85b6_ga10x_riscv";
}

