/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "inc/disp.h"
#include "disp/v02_02/dev_disp.h"
#include "g_disp_private.h"

LwU32
dispGetNumOrs_v02_02(LWOR orType)
{
    LwU32 val;

    val = GPU_REG_RD32(LW_PDISP_CLK_REM_MISC_CONFIGA);

    switch (orType)
    {
        case LW_OR_SOR:
            return DRF_VAL(_PDISP, _CLK_REM_MISC_CONFIGA, _NUM_SORS, val);
        case LW_OR_PIOR:
            return DRF_VAL(_PDISP, _CLK_REM_MISC_CONFIGA, _NUM_PIORS, val);
        case LW_OR_DAC:
            return DRF_VAL(_PDISP, _CLK_REM_MISC_CONFIGA, _NUM_DACS, val);
        default:
            dprintf("Error");
            return 0;
    }
}
