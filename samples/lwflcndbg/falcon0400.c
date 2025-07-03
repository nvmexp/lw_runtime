/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

/* ------------------------ Includes --------------------------------------- */
#include "maxwell/gm107/dev_falcon_v4.h"
#include "hal.h"
#include "falcon.h"

/*!
 *   
 * Returns register map based on falcon engine.
 *
 * @param[in]  engineBase   Specifies base address of the Falcon engine.
 * @param[in, out]  registerMap  FLCNGDB_REGISTER_MAP structure. 
 *
 */
void
flcnGetFlcngdbRegisterMap_v04_00
(
    LwU32                 engineBase,
    FLCNGDB_REGISTER_MAP* registerMap
)
{
    LwU32 i;

    registerMap->registerBase = engineBase;

    registerMap->icdCmd   = LW_PFALCON_FALCON_ICD_CMD + engineBase;
    registerMap->icdAddr  = LW_PFALCON_FALCON_ICD_ADDR + engineBase;
    registerMap->icdWData = LW_PFALCON_FALCON_ICD_WDATA + engineBase;
    registerMap->icdRData = LW_PFALCON_FALCON_ICD_RDATA + engineBase;

    registerMap->numBreakpoints = MAX_NUM_BREAKPOINTS;

    for (i = 0; i < MAX_NUM_BREAKPOINTS; i++)
    {
        registerMap->bpIBRK[i] = IBRKPT_REG_GET(i) + engineBase;
    }    
}

