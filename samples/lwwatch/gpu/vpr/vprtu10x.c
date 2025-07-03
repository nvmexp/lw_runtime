/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All information
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

#include "turing/tu102/dev_fuse.h"
#include "turing/tu102/dev_fb.h"

//-----------------------------------------------------
// vprIsActive_TU102
//-----------------------------------------------------
LwBool vprIsActive_TU102( LwU32 indexGpu)
{
    LwBool bVprActive          = LW_FALSE;
    LwU32 cmd                  = 0;

    cmd        = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_CYA_LO);
    bVprActive = DRF_VAL(_PFB, _PRI_MMU_VPR, _CYA_LO_IN_USE, cmd);

    return bVprActive;
}


//-----------------------------------------------------
// vprIsSupported_TU102
//-----------------------------------------------------
LwBool vprIsSupported_TU102( LwU32 indexGpu )
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
LwU32 vprGetFuseVersionScrubber_TU10X(LwU32 indexGpu)
{

    LwU32  fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_SCRUBBER_BIN_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_SCRUBBER_BIN_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersiolwprApp - Prints HW fuse version for vpr app
//-------------------------------------------------------------------
LwU32 vprGetFuseVersiolwprApp_TU10X(LwU32 indexGpu)
{
    LwU32  fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_PR_VPR_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_PR_VPR_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprGetFuseVersionSec2 - Prints HW fuse version for SEC2
//-------------------------------------------------------------------
LwU32 vprGetFuseVersionSec2_TU10X(LwU32 indexGpu)
{
    LwU32 fuseVersionHW = DRF_VAL(_FUSE, _OPT_FUSE_UCODE_SEC2_REV, _DATA, GPU_REG_RD32(LW_FUSE_OPT_FUSE_UCODE_SEC2_REV));

    return fuseVersionHW;
}

//-------------------------------------------------------------------
// vprPrintMemLockStatus - Prints global memory lock status
//-------------------------------------------------------------------
void vprPrintMemLockStatus_TU10X(LwU32 indexGpu)
{
    LwU32 fuse;

    fuse = GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_MODE);
    if(FLD_TEST_DRF(_PFB, _PRI_MMU_VPR_MODE, _VPR_PROTECTED_MODE, _TRUE, fuse))
    {
        if(FLD_TEST_DRF(_PFB, _PRI_MMU_VPR_MODE, _MEMORY_LOCKED, _TRUE, fuse))
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
LW_STATUS vprMemLockRangeInfo_TU10X(LwU32 indexGpu)
{
    LW_STATUS status                    = LW_OK;
    LwU64 vprMemLockStartAddressInBytes = 0;
    LwU64 vprMemLockEndAddressInBytes   = 0;
    LwU64 cmd                           = 0;

    // Read start address of MEM LOCK Region
    cmd   = REF_VAL(LW_PFB_PRI_MMU_LOCK_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_LOCK_ADDR_LO));
    vprMemLockStartAddressInBytes = cmd << LW_PFB_PRI_MMU_LOCK_ADDR_LO_ALIGNMENT;

    // Read End address of MEM LOCK Region
    cmd   = REF_VAL(LW_PFB_PRI_MMU_LOCK_ADDR_HI_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_LOCK_ADDR_HI));
    vprMemLockEndAddressInBytes   = cmd << LW_PFB_PRI_MMU_LOCK_ADDR_HI_ALIGNMENT;
    vprMemLockEndAddressInBytes  |= (NUM_BYTES_IN_4_KB - 1);

    dprintf("   0x%010llx   ||  0x%010llx ||", vprMemLockStartAddressInBytes, vprMemLockEndAddressInBytes);

    return status;
}

//-------------------------------------------------------------------
// vprMmuLwrrentRangeInfo - Prints current vpr range in mmu
//-------------------------------------------------------------------
LW_STATUS vprMmuLwrrentRangeInfo_TU10X(LwU32 indexGpu)
{
    LW_STATUS status           = LW_OK;
    LwBool bVprActive          = LW_FALSE;
    LwU64 vprRangeStartInBytes = 0;
    LwU64 vprRangeEndInBytes   = 0;
    LwU64 cmd                  = 0;

    bVprActive = pVpr[indexGpu].vprIsActive(indexGpu);

    if (bVprActive)
    {
        // Read start address of VPR
        cmd   = REF_VAL(LW_PFB_PRI_MMU_VPR_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_ADDR_LO));
        vprRangeStartInBytes = cmd << LW_PFB_PRI_MMU_VPR_ADDR_LO_ALIGNMENT;

        // Read end address of VPR
        cmd   = REF_VAL(LW_PFB_PRI_MMU_VPR_ADDR_HI_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_VPR_ADDR_HI));
        vprRangeEndInBytes   = cmd << LW_PFB_PRI_MMU_VPR_ADDR_HI_ALIGNMENT;

        //
        // IN HW, VPR address is 1 MB aligned and last MB is inclusive. So when
        // HW shows end address = 0x100000, the real end address is 0x1FFFFF.
        //
        vprRangeEndInBytes  |= (NUM_BYTES_IN_1_MB - 1);

        dprintf("   0x%010llx || 0x%010llx ||", vprRangeStartInBytes, vprRangeEndInBytes);
    }
    else
    {
        dprintf("       VPR is NOT setup         ||");
        status = LW_ERR_GENERIC;
    }

    return status;
}

