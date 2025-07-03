/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//
// includes
//

/* ------------------------ Includes --------------------------------------- */
#include "kepler/gk104/dev_falcon_v4.h"
#include "falcon.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */

LwU32
falconTrpcGetMaxIdx_GK104
(
    LwU32    engineBase
)
{
    return DRF_VAL(_PFALCON, _FALCON_TRACEIDX, _MAXIDX,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEIDX));
}

LwU32
falconTrpcGetPC_GK104
(
    LwU32    engineBase,
    LwU32    idx,
    LwU32*   pCount
)
{
    FLCN_REG_WR32(LW_PFALCON_FALCON_TRACEIDX, idx);
    if (pCount != NULL)
    {
        *pCount = 0;
    }
    return DRF_VAL(_PFALCON, _FALCON_TRACEPC, _PC,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEPC));
}
