/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
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

#include "pascal/gp102/dev_fb.h"
#include "pascal/gp102/dev_fuse.h"
#include "pascal/gp102/dev_gc6_island.h"
#include "pascal/gp102/dev_gc6_island_addendum.h"
#include "pascal/gp102/dev_sec_pri.h"

//-----------------------------------------------------
// vprIsSupported_GP102
//-----------------------------------------------------
LwBool vprIsSupported_GP102( LwU32 indexGpu )
{
    LwBool bSupported = LW_TRUE;
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_FUSE_SPARE_BIT_13);
    
    if(!FLD_TEST_DRF(_FUSE, _SPARE_BIT_13, _DATA, _ENABLE, fuse))
    {
        dprintf("VPR: VPR SW Fuse is not enabled\n");
        bSupported = LW_FALSE;
    }

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
// vprMmuLwrrentRangeInfo - Prints current vpr range in mmu
//-------------------------------------------------------------------
LW_STATUS vprMmuLwrrentRangeInfo_GP102(LwU32 indexGpu)
{
    LW_STATUS status           = LW_OK;
    LwBool bVprActive          = LW_FALSE;
    LwU64 vprRangeStartInBytes = 0;
    LwU64 vprRangeEndInBytes   = 0;

    bVprActive = DRF_VAL(_PFB, _PRI_MMU_VPR_INFO, _CYA_LO_IN_USE, pVpr[indexGpu].vprReadVprInfo(LW_PFB_PRI_MMU_VPR_INFO_INDEX_CYA_LO, indexGpu));

    if (bVprActive)
    {
        vprRangeStartInBytes = ((LwU64)pVpr[indexGpu].vprReadVprInfo(LW_PFB_PRI_MMU_VPR_INFO_INDEX_ADDR_LO, indexGpu))<<LW_PFB_PRI_MMU_VPR_INFO_ADDR_ALIGNMENT;
        vprRangeEndInBytes   = ((LwU64)pVpr[indexGpu].vprReadVprInfo(LW_PFB_PRI_MMU_VPR_INFO_INDEX_ADDR_HI, indexGpu))<<LW_PFB_PRI_MMU_VPR_INFO_ADDR_ALIGNMENT;

        dprintf("   0x%010llx || 0x%010llx ||", vprRangeStartInBytes, vprRangeEndInBytes);
    }
    else
    {
        dprintf("       VPR is NOT setup         ||");
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-------------------------------------------------------------------
// vprBsiMaxRangeInfo - Prints max vpr range in bsi
//-------------------------------------------------------------------
LW_STATUS vprBsiMaxRangeInfo_GP102(LwU32 indexGpu)
{
    LW_STATUS status      = LW_OK;
    LwU64 vprSizeInMBytes = 0;

    vprSizeInMBytes = ((LwU64)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_13, _MAX_VPR_SIZE_MB));

    if (vprSizeInMBytes)
    {
        dprintf("     0x%04llx MB      ||", vprSizeInMBytes);
    }
    else
    {
        dprintf(" VPR is NOT setup   ||");
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-------------------------------------------------------------------
// vprBsiLwrrentRangeInfo - Prints current vpr range in bsi
//-------------------------------------------------------------------
LW_STATUS vprBsiLwrrentRangeInfo_GP102(LwU32 indexGpu)
{
    LW_STATUS status             = LW_OK;
    LwU64 vprStartAddressInBytes = 0;
    LwU64 vprEndAddressInBytes   = 0;
    LwU64 vprSizeInBytes         = 0;

    vprSizeInBytes = ((LwU64)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_13, _LWRRENT_VPR_SIZE_MB)) << SHIFT_1MB;

    if (vprSizeInBytes)
    {
        // Read start address of current vpr range and change to byte alignment.
        vprStartAddressInBytes = ((LwU64)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_14, _LWRRENT_VPR_RANGE_START_ADDR_MB_ALIGNED)) << SHIFT_1MB;

        // Callwlate end address based on start address and current vpr size
        vprEndAddressInBytes = vprStartAddressInBytes + vprSizeInBytes - 1;

        dprintf("  0x%010llx  || 0x%010llx ||", vprStartAddressInBytes, vprEndAddressInBytes);
    }
    else
    {
        dprintf("       VPR is NOT enabled       ||");
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-------------------------------------------------------------------
// vprPrintMemLockStatus - Prints global memory lock status
//-------------------------------------------------------------------
void vprPrintMemLockStatus_GP102(LwU32 indexGpu)
{
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_WPR_WRITE);
    if(FLD_TEST_DRF(_PFB, _PRI_MMU_VPR_WPR_WRITE, _VPR_PROTECTED_MODE, _TRUE, fuse))
    {
        if(FLD_TEST_DRF(_PFB, _PRI_MMU_VPR_WPR_WRITE, _VPR_MEMORY_LOCKED, _TRUE, fuse))
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
// vprMemLockRangeInfo - Prints memory lock range
//-------------------------------------------------------------------
LW_STATUS vprMemLockRangeInfo_GP102(LwU32 indexGpu)
{
    LW_STATUS status                    = LW_OK;
    LwU64 vprMemLockStartAddressInBytes = 0;
    LwU64 vprMemLockEndAddressInBytes   = 0;

    vprMemLockStartAddressInBytes = ((LwU64)pVpr[indexGpu].vprReadWprInfo(LW_PFB_PRI_MMU_WPR_INFO_INDEX_LOCK_ADDR_LO, indexGpu)) << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;
    vprMemLockEndAddressInBytes   = ((LwU64)pVpr[indexGpu].vprReadWprInfo(LW_PFB_PRI_MMU_WPR_INFO_INDEX_LOCK_ADDR_HI, indexGpu)) << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;

    dprintf("   0x%010llx   ||  0x%010llx ||", vprMemLockStartAddressInBytes, vprMemLockEndAddressInBytes);

    return status;
}

//--------------------------------------------------------------------
// vprPrintBsiType1LockStatus - Prints type1 lock status of BSI Scratch
//--------------------------------------------------------------------
void vprPrintBsiType1LockStatus_GP102(LwU32 indexGpu)
{
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_PGC6_BSI_VPR_SELWRE_SCRATCH_14);
    if(FLD_TEST_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_14, _HDCP22_TYPE1_LOCK, _ENABLE, fuse))
    {
        dprintf("            LOCKED            ||");
    }
    else
    {
        dprintf("          NOT LOCKED          ||");
    }
}

//-------------------------------------------------------------------
// vprGetFuseVersionAcr - Prints HW fuse version for acr
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionAcr_GP102(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_ACR_HS_REV);

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetUcodeVersionAcr - Prints ucode version for acr
//-------------------------------------------------------------------
LwU32 vprGetUcodeVersionAcr_GP102(LwU32 indexGpu)
{
    LwU32 ucodeVersion = ((LwU32)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_14, _ACR_BINARY_VERSION));

    return ucodeVersion;
}

//-------------------------------------------------------------------
// vprGetFuseVersionCtxsw - Prints HW fuse version for ctxsw
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionCtxsw_GP102(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_7) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_8) & 0x1;
    LwU32  fuseVersionHW = bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionLwdec - Prints HW fuse version for lwdec
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionLwdec_GP102(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_LWDEC_REV);

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionScrubber - Prints HW fuse version for scrubber
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionScrubber_GP102(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_5) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_6) & 0x1;
    LwU32  bit2 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_4) & 0x1;
    LwU32  fuseVersionHW = bit2 << 2 | bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetUcodeVersionScrubber - Prints ucode version for scrubber
//-------------------------------------------------------------------
LwU32 vprGetUcodeVersionScrubber_GP102(LwU32 indexGpu)
{
    LwU32 ucodeVersion = ((LwU32)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_15, _SCRUBBER_BINARY_VERSION));

    return ucodeVersion;
}

//-------------------------------------------------------------------
// vprGetFuseVersionSec2 - Prints HW fuse version for SEC2
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionSec2_GP102(LwU32 indexGpu)
{
    LwU32 bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_0) & 0x1;
    LwU32 bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_1) & 0x1;
    LwU32 bit2 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_2) & 0x1;
    LwU32 fuseVersionHW = bit2 << 2 | bit1 << 1 | bit0;

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionUde - Prints fuse version for ude
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionUde_GP102(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_UDE_REV);

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetUcodeVersionUde - Prints ucode version for ude
//-------------------------------------------------------------------
LwU32 vprGetUcodeVersionUde_GP102(LwU32 indexGpu)
{
    LwU32 ucodeVersion = ((LwU32)GPU_REG_RD_DRF(_PGC6, _BSI_VPR_SELWRE_SCRATCH_15, _VBIOS_UDE_VERSION));

    return ucodeVersion;
}

//-------------------------------------------------------------------
// vprGetFuseVersiolwprApp - Prints HW fuse version for vpr app
//-------------------------------------------------------------------
LwU32 vprGetFuseVersiolwprApp_GP102(LwU32 indexGpu)
{
    LwU32  bit0 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_14) & 0x1;
    LwU32  bit1 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_15) & 0x1;
    LwU32  bit2 = GPU_REG_RD32(LW_FUSE_SPARE_BIT_16) & 0x1;
    LwU32  fuseVersionHW = bit2 << 2 | bit1 << 1 | bit0;

    return fuseVersionHW;
}


/*!
 * @brief: VPR info can be read by querying VPR_INFO registers in MMU
 */
LwU32 vprReadVprInfo_GP102(LwU32 index, LwU32 indexGpu)
{
    LwU32 cmd = 0;

    while(LW_TRUE)
    {
        cmd = DRF_NUM(_PFB, _PRI_MMU_VPR_INFO, _INDEX, index);
        GPU_REG_WR32(LW_PFB_PRI_MMU_VPR_INFO, cmd);

        cmd = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_INFO);

        // Ensure that VPR info that we read has correct index.
        if (FLD_TEST_DRF_NUM(_PFB, _PRI_MMU_VPR_INFO, _INDEX, index, cmd))
        {
            LwU32 val = DRF_VAL(_PFB, _PRI_MMU_VPR_INFO, _DATA, cmd);

            if (index == LW_PFB_PRI_MMU_VPR_INFO_INDEX_CYA_LO ||
                index == LW_PFB_PRI_MMU_VPR_INFO_INDEX_CYA_HI)
            {
                // Basically left shift by 4 bits and return value of read/write mask.
                return DRF_NUM(_PFB, _PRI_MMU_VPR_INFO, _DATA, val);
            }
            else if (index == LW_PFB_PRI_MMU_VPR_INFO_INDEX_ADDR_LO ||
                     index == LW_PFB_PRI_MMU_VPR_INFO_INDEX_ADDR_HI)
            {
                 return val;
            }
        }
    }
}

/*!
 * @brief: WPR info can be read by querying WPR_INFO registers in MMU
 */
LwU32 vprReadWprInfo_GP102(LwU32 index, LwU32 indexGpu)
{
    LwU32 cmd = 0;

    while(LW_TRUE)
    {
        cmd = DRF_NUM(_PFB, _PRI_MMU_WPR_INFO, _INDEX, index);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);

        cmd = GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_INFO);

        // Ensure that VPR info that we read has correct index.
        if (FLD_TEST_DRF_NUM(_PFB, _PRI_MMU_WPR_INFO, _INDEX, index, cmd))
        {
            LwU32 val = DRF_VAL(_PFB, _PRI_MMU_WPR_INFO, _DATA, cmd);

            if (index == LW_PFB_PRI_MMU_WPR_INFO_INDEX_LOCK_ADDR_LO ||
                index == LW_PFB_PRI_MMU_WPR_INFO_INDEX_LOCK_ADDR_HI)
            {
                return val;
            }
        }
    }
}
