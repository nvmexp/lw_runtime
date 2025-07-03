/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2013 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "maxwell/gm200/dev_top.h"
#include "maxwell/gm200/dev_ce_pri.h"

#include "hal.h"

#include "kepler/kepler_ce.h"

//-----------------------------------------------------
// ceGetPrivs_GM200 - Returns the CE priv reg space
//-----------------------------------------------------
void *ceGetPrivs_GM200( void )
{
    static dbg_ce cePrivReg[] =
    {
        privInfo_ce(LW_PCE_COP_CMD0),
        privInfo_ce(LW_PCE_COP_LINES_TO_COPY),
        privInfo_ce(LW_PCE_COP_SRC_PARAM0),
        privInfo_ce(LW_PCE_COP_DST_PARAM0),
        privInfo_ce(LW_PCE_COP_SRC_PARAM1),
        privInfo_ce(LW_PCE_COP_DST_PARAM1),
        privInfo_ce(LW_PCE_COP_SRC_PARAM2),
        privInfo_ce(LW_PCE_COP_DST_PARAM2),
        privInfo_ce(LW_PCE_COP_SRC_PARAM3),
        privInfo_ce(LW_PCE_COP_DST_PARAM3),
        privInfo_ce(LW_PCE_COP_SRC_PARAM4),
        privInfo_ce(LW_PCE_COP_DST_PARAM4),
        privInfo_ce(LW_PCE_COP_SWIZZLE_CONSTANT0),
        privInfo_ce(LW_PCE_COP_SWIZZLE_CONSTANT1),
        privInfo_ce(LW_PCE_COP_CMD1),
        privInfo_ce(LW_PCE_COP_BIND),
        privInfo_ce(LW_PCE_COP_SRC_PHYS_MODE),
        privInfo_ce(LW_PCE_COP_DST_PHYS_MODE),
        privInfo_ce(LW_PCE_COP2_PIPESTATUS),
        privInfo_ce(LW_PCE_COP2_INTR_EN),
        privInfo_ce(LW_PCE_COP2_INTR_STATUS),
        privInfo_ce(LW_PCE_PMM),
        privInfo_ce(LW_PCE_ENGCAP),
        privInfo_ce(LW_PCE_FE_BORROW),
        privInfo_ce(LW_PCE_FE_INJECT_DMA),
        privInfo_ce(LW_PCE_FE_INJECT_BIND),
        privInfo_ce(LW_PCE_FE_BIND_STATUS),
        privInfo_ce(LW_PCE_FE_IRQ),
        privInfo_ce(LW_PCE_FE_LAUNCHERR),
        privInfo_ce(LW_PCE_FE_ENGCTL),
        privInfo_ce(LW_PCE_FE_THROTTLE),
        privInfo_ce(0)
    };

    return (void *)cePrivReg;
}
