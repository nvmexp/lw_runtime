
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
#include "lwtypes.h"

#include "pascal/gp100/dev_fb.h"
#include "pascal/gp100/dev_master.h"
#include "pascal/gp100/dev_falcon_v4.h"

#include "g_falcphys_private.h"

void falcphysGetMmuPhysRegConfig_GP100 (void *pFalcVoid)
{
    FALCMMUPHYS *pFalcMmu = (FALCMMUPHYS *)pFalcVoid;
    pFalcMmu->regBase = LW_PFB_PRI_MMU_PHYS_SELWRE;
    pFalcMmu->fecsInit = LW_PFB_PRI_MMU_PHYS_SELWRE_FECS_INIT;
    pFalcMmu->pmuInit = LW_PFB_PRI_MMU_PHYS_SELWRE_PMU_INIT;
    pFalcMmu->secInit = LW_PFB_PRI_MMU_PHYS_SELWRE_SEC_INIT;
    pFalcMmu->dfalconInit = LW_PFB_PRI_MMU_PHYS_SELWRE_DFALCON_INIT;
    pFalcMmu->afalconInit = LW_PFB_PRI_MMU_PHYS_SELWRE_AFALCON_INIT;
    pFalcMmu->lwdecInit = LW_PFB_PRI_MMU_PHYS_SELWRE_LWDEC_INIT;
    pFalcMmu->lwencInit = LW_PFB_PRI_MMU_PHYS_SELWRE_LWENC_INIT;
    pFalcMmu->mspppInit = LW_PFB_PRI_MMU_PHYS_SELWRE_MSPPP_INIT;
}

void falcphysGetLwenc2FalcPhysProp_GP100 (FALCONPROP *pFalc)
{
    pFalc[LWENC2_FALCON_ID].regBase       = LW_FALCON_LWENC2_BASE;
    pFalc[LWENC2_FALCON_ID].pmcMask       = DRF_DEF(_PMC, _ENABLE, _LWENC2, _ENABLED);
}

void falcphysProgramDmaBase1Reg_GP100 (FALCONPROP *pFalc, LwU32 falconId, LwU64 addr)
{
    GPU_REG_WR32((pFalc[falconId].regBase + LW_PFALCON_FALCON_DMATRFBASE1), LwU64_HI32(addr));
}
