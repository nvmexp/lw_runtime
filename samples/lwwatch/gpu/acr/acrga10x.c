/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "acr.h"
#include "rmlsfm.h"
#include "g_acr_private.h"  

#include "ampere/ga102/dev_top.h"
#include "ampere/ga102/dev_fb.h"
#include "ampere/ga102/dev_fuse.h"
#include "ampere/ga102/dev_master.h"

//-----------------------------------------------------
// acrIsSec2FalconEnabled_GA10X
//-----------------------------------------------------
LwBool acrIsSec2FalconEnabled_GA10X()
{
     LwU32 deviceResetBit   = (LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_SEC0) % 32;
     LwU32 deviceIndexBit   = (LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_SEC0) / 32;

     return FLD_IDX_TEST_DRF(_PMC, _DEVICE_ENABLE, _STATUS_BIT, deviceResetBit, _ENABLE, GPU_REG_RD32(LW_PMC_DEVICE_ENABLE(deviceIndexBit)));
}

//-----------------------------------------------------
// acrGetDpuCfgaSize_GA10X
//-----------------------------------------------------
LwU32 acrGetDpuCfgaSize_GA10X()
{
     return LW_PFB_PRI_MMU_FALCON_GSP_CFGA__SIZE_1;
}


BOOL acrIsFmodel_GA10X (void)
{
    //
    // These fuses are blown in all cases on silicon, so if they are not,
    // then we are either running on simulation/emulation
    //
    if ((DRF_VAL(_FUSE, _OPT_LOT_CODE_0, _DATA, GPU_REG_RD32(LW_FUSE_OPT_LOT_CODE_0))
        || DRF_VAL(_FUSE, _OPT_LOT_CODE_1, _DATA, GPU_REG_RD32(LW_FUSE_OPT_LOT_CODE_1))
        || DRF_VAL(_FUSE, _OPT_FAB_CODE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FAB_CODE))
        || DRF_VAL(_FUSE, _OPT_X_COORDINATE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_X_COORDINATE))
        || DRF_VAL(_FUSE, _OPT_Y_COORDINATE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_Y_COORDINATE))
        || DRF_VAL(_FUSE, _OPT_WAFER_ID, _DATA, GPU_REG_RD32(LW_FUSE_OPT_WAFER_ID))
        || DRF_VAL(_FUSE, _OPT_VENDOR_CODE, _DATA, GPU_REG_RD32(LW_FUSE_OPT_VENDOR_CODE)))
        == 0)
    {
        // To check if its emulation of simulation (fmodel)
        if (DRF_VAL(_PMC, _BOOT_2, _EMULATION, GPU_REG_RD32(LW_PMC_BOOT_2)) == LW_PMC_BOOT_2_EMULATION_NO)
        {
            return TRUE;
        }
    }
    return FALSE;
}


