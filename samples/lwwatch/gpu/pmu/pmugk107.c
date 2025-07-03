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
// lwwatch PMU helper.  
// pmugk107.c
//
//*****************************************************

//
// includes
//
#include "falcon.h"
#include "kepler/gk107/dev_pwr_pri.h"

void
pmuFlcngdbGetRegMap_GK107
(
    FLCNGDB_REGISTER_MAP* registerMap
)
{
    flcnGetFlcngdbRegisterMap_v04_00(LW_PPWR_FALCON_IRQSSET, registerMap);
}

const char *
pmuUcodeName_GK107()
{
    return "g_c85b6_gk10x";
}
