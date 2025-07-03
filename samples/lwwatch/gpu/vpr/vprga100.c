/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "vpr.h"
#include "rmlsfm.h"
#include "os.h"
#include "mmu/mmucmn.h"

#include "g_vpr_private.h"          // (rmconfig) hal/obj setup

#include "ampere/ga100/dev_fuse.h"

//-----------------------------------------------------
// vprIsSupported_GA100
//-----------------------------------------------------
LwBool vprIsSupported_GA100( LwU32 indexGpu )
{
    LwBool bSupported = LW_TRUE;
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_FUSE_OPT_VPR_ENABLED);

    if(!FLD_TEST_DRF(_FUSE, _OPT_VPR_ENABLED, _DATA, _YES, fuse))
    {
        dprintf("VPR: VPR HW Fuse is not blown\n");
        bSupported = LW_FALSE;
    }

    if(!bSupported)
    {
        dprintf("\lw: VPR not supported on GPU %d.\n\n", indexGpu);
    }

    return bSupported;
}

//-------------------------------------------------------------------
// vprGetFuseVersionScrubber - Prints HW fuse version for scrubber
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionScrubber_GA100(LwU32 indexGpu)
{

    LwU32  fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_SCRUBBER_BIN_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_SCRUBBER_BIN_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersiolwprApp - Prints HW fuse version for vpr app
//-------------------------------------------------------------------
LwU32 vprGetFuseVersiolwprApp_GA100(LwU32 indexGpu)
{
    LwU32  fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_PR_VPR_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_PR_VPR_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionSec2 - Prints HW fuse version for SEC2
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionSec2_GA100(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_SEC2_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_SEC2_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionAcr - Prints HW fuse version for acr
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionAcr_GA100(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_ACR_HS_REV);

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionCtxsw - Prints HW fuse version for ctxsw
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionCtxsw_GA100(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_7) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_8) & 0x1;
    LwU32  fuseVersionHW = bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionLwdec - Prints HW fuse version for lwdec
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionLwdec_GA100(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_LWDEC_REV);

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionUde - Prints fuse version for ude
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionUde_GA100(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_GFW_REV);

    return fuseVersionHW;
}

