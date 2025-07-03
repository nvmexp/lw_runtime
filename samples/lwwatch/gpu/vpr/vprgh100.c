/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "vpr.h"

#include "g_vpr_private.h"          // (rmconfig) hal/obj setup

#include "hopper/gh100/dev_fb.h"
#include "hopper/gh100/dev_fuse.h"

//-----------------------------------------------------
// vprIsActive_GH100
//-----------------------------------------------------
LwBool vprIsActive_GH100( LwU32 indexGpu)
{
    LwBool bVprActive          = LW_FALSE;
    LwU32 cmd                  = 0;

    cmd        = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_MODE);
    bVprActive = DRF_VAL(_PFB, _PRI_MMU_VPR, _MODE_IN_USE, cmd);

    return bVprActive;
}

//-------------------------------------------------------------------
// vprPrintMemLockStatus - Prints global memory lock status
//-------------------------------------------------------------------
void vprPrintMemLockStatus_GH100(LwU32 indexGpu)
{
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_STATE);
    if(FLD_TEST_DRF(_PFB, _PRI_MMU_VPR_STATE, _PROTECTED_MODE, _TRUE, fuse))
    {
        if(FLD_TEST_DRF(_PFB, _PRI_MMU_MEMORY_LOCKED, _ENABLE, _TRUE, fuse))
        {
            dprintf("        LOCKED        ||");
        }
        else
        {
            dprintf("      NOT LOCKED      ||");
        }
    }
    else
    {
        dprintf("   UNPROTECTED MODE   ||");
    }
}

//-------------------------------------------------------------------
// vprGetFuseVersionCtxsw - Prints HW fuse version for ctxsw
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionCtxsw_GH100(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_7) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_8) & 0x1;
    LwU32  fuseVersionHW = bit1 << 1 | bit0;

    return fuseVersionHW;
}


