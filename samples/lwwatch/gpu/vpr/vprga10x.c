/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "vpr.h"

#include "g_vpr_private.h"          // (rmconfig) hal/obj setup

#include "ampere/ga102/dev_fuse.h"

//-------------------------------------------------------------------
// vprGetFuseVersionCtxsw - Prints HW fuse version for ctxsw
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionCtxsw_GA10X(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_7) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_8) & 0x1;
    LwU32  fuseVersionHW = bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-----------------------------------------------------
// vprIsSupported_GA10X
//-----------------------------------------------------
LwBool vprIsSupported_GA10X( LwU32 indexGpu )
{
    LwBool bSupported = LW_TRUE;
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_FUSE_OPT_SW_VPR_ENABLED);

    if(!FLD_TEST_DRF(_FUSE, _OPT_SW_VPR_ENABLED, _DATA, _YES, fuse))
    {
        dprintf("VPR: VPR SW Fuse is not enabled\n");
        bSupported = LW_FALSE;
        return bSupported;
    }

    return vprIsSupported_GA100(indexGpu);
}
