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

#include "volta/gv100/dev_fuse.h"
#include "disp/v03_00/dev_disp.h"

//-------------------------------------------------------------------
// vprGetFuseVersiolwprApp - Prints HW fuse version for vpr app
//-------------------------------------------------------------------
LwU32 vprGetFuseVersiolwprApp_GV100(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_14) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_15) & 0x1;
    LwU32  bit2 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_16) & 0x1;
    LwU32  fuseVersionHW = bit2 << 2 | bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionUde - Prints fuse version for ude
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionUde_GV100(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_GFW_REV);

    return fuseVersionHW;
}

//--------------------------------------------------------------------
// vprPrintBsiType1LockStatus - Prints type1 lock status of BSI Scratch
//--------------------------------------------------------------------
void vprPrintBsiType1LockStatus_GV100(LwU32 indexGpu)
{
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_PDISP_UPSTREAM_HDCP_VPR_POLICY_CTRL);
    if(FLD_TEST_DRF(_PDISP_UPSTREAM, _HDCP_VPR_POLICY_CTRL, _BLANK_VPR_ON_HDCP22_TYPE1, _ENABLE, fuse))
    {
        dprintf("            LOCKED            ||");
    }
    else
    {
        dprintf("          NOT LOCKED          ||");
    }
}

