/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "falcphys.h"
#include "exts.h"
#include "g_falcphys_hal.h"

//-------------------------------------------------------------------
// falcphysIsSupported - Determines if falcphysdmacheck is supported
//-------------------------------------------------------------------
BOOL falcphysIsSupported(LwU32 indexGpu)
{
    if (!pFalcphys[indexGpu].falcphysIsSupported(indexGpu))
    {
        dprintf("lw: falcphysdmacheck not supported on GPU %d.\n", indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: falcphysdmacheck supported on GPU %d.\n", indexGpu);
        return TRUE;
    }
}

//-------------------------------------------------------------------
// falcphysDmaAccessCheck - Verify falcon physical DMA access restrictions
//-------------------------------------------------------------------
LW_STATUS falcphysDmaAccessCheck(LwU32 indexGpu)
{
    LW_STATUS status;

    if (!pFalcphys[indexGpu].falcphysIsSupported(indexGpu))
    {
        dprintf("lw: falcphysLsfStatus: falcphysdmacheck not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }

    dprintf("Verifies falcon physical DMA access restrictions\n");

    CHECK(pFalcphys[indexGpu].falcphysDmaAccessCheck(indexGpu))

    dprintf("\nSuccessful\n");

    return LW_OK;
}

//-----------------------------------------------------
// falcphysDisplayHelp - Display related help info
//-----------------------------------------------------
void falcphysDisplayHelp(void)
{
    dprintf(" falcphysdmacheck commands:\n");
    dprintf(" falcphys \"-help\"                   - Displays the ACR related help menu\n");
    dprintf(" falcphys \"-supported\"              - Determines if ACR is supported on available GPUs\n");
    dprintf(" falcphys \"-accesscheck\"            - Verifies falcon physical DMA access restrictions\n");
}
