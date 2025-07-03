/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "acr.h"
#include "rmlsfm.h"
#include "os.h"

#include "chip.h"
#include "disp.h"
#include "pmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "clk.h"
#include "smbpbi.h"
#include "falcphys.h"
#include "mmu/mmucmn.h"

#include "g_acr_private.h"          // (rmconfig) hal/obj setup

#include "ampere/ga100/dev_pwr_pri.h"
#include "ampere/ga100/dev_master.h"
#include "ampere/ga100/dev_fbfalcon_pri.h"
#include "ampere/ga100/ioctrl_discovery.h"
#include "ampere/ga100/dev_gsp.h"
#include "ampere/ga100/dev_lwdec_pri.h"
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/dev_falcon_v4.h"
#include "ampere/ga100/dev_graphics_nobundle.h"
#include "ampere/ga100/dev_sec_pri.h"
#include "ampere/ga100/dev_top.h"
#include "ampere/ga100/dev_gc6_island.h"
#include "ampere/ga100/dev_gc6_island_addendum.h"
#include "ampere/ga100/dev_fuse.h"

/* ------------------------- Macros and Defines ----------------------------- */
#define LW_PMC_DEVICE_RESET_BIT(device)     ((LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_##device) % 32)
#define LW_PMC_DEVICE_INDEX_BIT(device)     ((LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_##device) / 32)

/* ------------------------- Static Functions ------------------------------ */

void acrGetFalconProp_GA100(LSFALCPROP *pFalc, LwU32  falconId, LwU32 indexGpu)
{
    switch(falconId)
    {
        case LSF_FALCON_ID_PMU:
                pFalc[falconId].name            = "PMU   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_PWR_BASE;
                pFalc[falconId].bFalconEnabled  = TRUE;
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_PMU_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_PMU_CFGB(0);
                pFalc[falconId].regCfgPLM      = LW_PFB_PRI_MMU_PMU_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_PMU_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_DPU:
                pFalc[falconId].name            = "GSP   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PGSP);
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PDISP), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_GSP_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_GSP_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_GSP_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = pAcr[indexGpu].acrGetDpuCfgaSize();
                break;
        case LSF_FALCON_ID_FECS:
                pFalc[falconId].name            = "FECS  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_FECS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _DEVICE_ENABLE, _STATUS_BIT, LW_PMC_DEVICE_RESET_BIT(GRAPHICS0),
                                                                   _ENABLE, GPU_REG_RD32(LW_PMC_DEVICE_ENABLE(LW_PMC_DEVICE_INDEX_BIT(GRAPHICS0))));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_FECS_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_FECS_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_FECS_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_FECS_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_GPCCS:
                pFalc[falconId].name            = "GPCCS ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_GPC0_GPCCS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _DEVICE_ENABLE, _STATUS_BIT, LW_PMC_DEVICE_RESET_BIT(GRAPHICS0),
                                                                   _ENABLE, GPU_REG_RD32(LW_PMC_DEVICE_ENABLE(LW_PMC_DEVICE_INDEX_BIT(GRAPHICS0))));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_GPCCS_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWDEC:
                pFalc[falconId].name            = "LWDEC0";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWDEC_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _DEVICE_ENABLE, _STATUS_BIT, LW_PMC_DEVICE_RESET_BIT(LWDEC0),
                                                                   _ENABLE, GPU_REG_RD32(LW_PMC_DEVICE_ENABLE(LW_PMC_DEVICE_INDEX_BIT(LWDEC0))));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_LWDEC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWENC0:
                pFalc[falconId].name            = "LWENC0";
                pFalc[falconId].available       = LW_FALSE;
                pFalc[falconId].bFalconEnabled  = LW_FALSE;
                break;
        case LSF_FALCON_ID_LWENC1:
                pFalc[falconId].name            = "LWENC1";
                pFalc[falconId].available       = LW_FALSE;
                pFalc[falconId].bFalconEnabled  = LW_FALSE;
                break;
        case LSF_FALCON_ID_SEC2:
                pFalc[falconId].name            = "SEC2  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PSEC_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = pAcr[indexGpu].acrIsSec2FalconEnabled();
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_SEC_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_SEC_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_SEC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_SEC_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWENC2:
                pFalc[falconId].name            = "LWENC2";
                pFalc[falconId].available       = LW_FALSE;
                break;
        case LSF_FALCON_ID_MINION:
                pFalc[falconId].name            = "MINION";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_MINION;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWLINK), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = 0;
                pFalc[falconId].regBaseCfgb     = 0;
                pFalc[falconId].regCfgPLM       = 0;
                pFalc[falconId].size            = 0;
                break;
        case LSF_FALCON_ID_FBFALCON:
                pFalc[falconId].name            = "FBFALCON";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PFBFALCON);
                pFalc[falconId].bFalconEnabled  = LW_TRUE;
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_FBFALCON_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_FBFALCON_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_FBFALCON_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_FBFALCON_CFGA__SIZE_1;
                break;
        default:
                break;
    }
}

//-----------------------------------------------------
// acrIsSupported_GA100
//-----------------------------------------------------
BOOL acrIsSupported_GA100( LwU32 indexGpu )
{
    LwU32 fuse;
    LwU32 falconId;

    // Populate falcon data
    for (falconId=0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        pAcr[indexGpu].acrGetFalconProp(lsFalc, falconId, indexGpu);
    }

    // Skip IsSupported check in case of fmodel
    if (pAcr[indexGpu].acrIsFmodel())
    {
        return TRUE;
    }

    // Check the PRIV SEC fuse
    fuse = GPU_REG_RD32(LW_FUSE_OPT_PRIV_SEC_EN);
    if(!FLD_TEST_DRF(_FUSE, _OPT_PRIV_SEC_EN, _DATA, _YES, fuse))
    {
        dprintf("ACR: PRIVSEC is not enabled\n");
        return FALSE;
    }

    fuse = GPU_REG_RD32(LW_FUSE_OPT_WPR_ENABLED);
    if(!FLD_TEST_DRF(_FUSE, _OPT_WPR_ENABLED, _DATA, _YES, fuse))
    {
        dprintf("ACR: WPR fuse is not blown\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// acrIsSec2FalconEnabled_GA100
//-----------------------------------------------------
LwBool acrIsSec2FalconEnabled_GA100()
{
     LwU32 deviceResetBit   = (LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_SEC0) % 32;
     LwU32 deviceIndexBit   = (LW_PTOP_DEVICE_INFO_RESET_BIT_FOR_UCODE_AND_VBIOS_ONLY_SEC0) / 32;

     return FLD_IDX_TEST_DRF(_PMC, _DEVICE_ENABLE, _STATUS_BIT, deviceResetBit, _ENABLE, GPU_REG_RD32(LW_PMC_DEVICE_ENABLE(deviceIndexBit)));
}

//-----------------------------------------------------
// acrGetDpuCfgaSize_GA100
//-----------------------------------------------------
LwU32 acrGetDpuCfgaSize_GA100()
{
     return LW_PFB_PRI_MMU_FALCON_GSP_CFGA__SIZE_1;
}
