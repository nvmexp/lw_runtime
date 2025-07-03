/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All information
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
#include "g_fb_hal.h"

#include "turing/tu102/dev_pwr_pri.h"
#include "turing/tu102/dev_master.h"
#include "turing/tu102/dev_fbfalcon_pri.h"
#include "turing/tu102/dev_minion.h"
#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_lwenc_pri_sw.h"
#include "turing/tu102/dev_lwdec_pri.h"
#include "turing/tu102/dev_fb.h"
#include "turing/tu102/dev_falcon_v4.h"
#include "turing/tu102/dev_graphics_nobundle.h"
#include "turing/tu102/dev_sec_pri.h"
#include "dpu/v02_05/dev_disp_falcon.h"
#include "turing/tu102/dev_gc6_island.h"
#include "turing/tu102/dev_gc6_island_addendum.h"

/* ------------------------- Macros and Defines ----------------------------- */
#define LW_MWPR_DEFAULT_ALIGNMENT           (0X1000)
#define LW_WPR_EXPECTED_READ_MASK           (0x08)
#define LW_WPR_EXPECTED_WRITE_MASK          (0x08)
#define LW_SUBWPR_EXPECTED_WRITE_MASK       (0x0C)
#define LW_SUBWPR_EXPECTED_READ_MASK        (0x0C)

//
// Setting default alignment to 128KB since WPR addresses
// stored in MMU are 128K aligned.
// TODO:Replace this with manual define when its available
//
#define LW_ACR_DEFAULT_ALIGNMENT            (0x20000)

#define ACR_WPR1_REGION_IDX                 (0x0)
#define ACR_WPR2_REGION_IDX                 (0x1)


/* ------------------------- Static Functions ------------------------------ */
static LW_STATUS _printFlcnWprInfo(LwU32 regionId, LwU64 startAddr, LwU64 endAddr, LwU32 readMask, LwU32 writeMask);
static LW_STATUS _acrGetSelwreScratchAddrForSubWpr(LwU32 falconId, LwU32 flcnSubWprId, LwU32 *pScratchCfga, LwU32 *pScratchCfgb, LwU32 *pScratchPlm);
LW_STATUS acrVerifySharedSubWpr(LwU32 falconId, LwU8 flcnSubWprId, LwU8 readMaskExp, LwU8 writeMaskExp, LwU32 *size);


/* ------------------------- Global Variables ------------------------------ */
LwU64   startAddr[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1];
LwU64   endAddr[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1];
LwU64   regionSize[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1];
LwU32   readMask[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1];
LwU32   writeMask[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1];

void acrGetFalconProp_TU10X(LSFALCPROP *pFalc, LwU32  falconId, LwU32 indexGpu)
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
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_GSP_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_FECS:
                pFalc[falconId].name            = "FECS  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_FECS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_FECS_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_FECS_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_FECS_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_FECS_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_GPCCS:
                pFalc[falconId].name            = "GPCCS ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_GPC0_GPCCS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_GPCCS_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_GPCCS_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWDEC:
                pFalc[falconId].name            = "LWDEC0";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWDEC_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWDEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_LWDEC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_LWDEC0_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWENC0:
                pFalc[falconId].name            = "LWENC0";
                pFalc[falconId].regBase         = LW_FALCON_LWENC0_BASE;
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC0), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_LWENC0_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_LWENC0_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_LWENC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_LWENC0_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWENC1:
                pFalc[falconId].name            = "LWENC1";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWENC1_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC1), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_LWENC1_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_LWENC1_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_LWENC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_LWENC1_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_SEC2:
                pFalc[falconId].name            = "SEC2  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PSEC_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_SEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_SEC_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_SEC_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_SEC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_SEC_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_LWENC2:
                pFalc[falconId].name            = "LWENC2";
                pAcr[indexGpu].acrGetLwenc2FalconProp(pFalc);
                pFalc[falconId].regBaseCfga     = LW_PFB_PRI_MMU_FALCON_LWENC2_CFGA(0);
                pFalc[falconId].regBaseCfgb     = LW_PFB_PRI_MMU_FALCON_LWENC2_CFGB(0);
                pFalc[falconId].regCfgPLM       = LW_PFB_PRI_MMU_LWENC_PRIV_LEVEL_MASK;
                pFalc[falconId].size            = LW_PFB_PRI_MMU_FALCON_LWENC2_CFGA__SIZE_1;
                break;
        case LSF_FALCON_ID_MINION:
                pFalc[falconId].name            = "MINION";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PMINION_FALCON);
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

//------------------------------------------------------------
// acrGetMultipleWprInfo_TU10X - Get ACR region info
//------------------------------------------------------------
LW_STATUS acrGetMultipleWprInfo_TU10X( LwU32 indexGpu)
{
    LwU32      falconId;
    LwU64      startAddr;
    LwU64      endAddr;
    LwU32      readMask;
    LwU32      writeMask;
    LwU32      index;
    LwU32      valCfga;
    LwU32      valCfgb;

    dprintf("mWPR info according to falcons\n");

    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("|| FalconId ||No. of Regions||Region Index|| StartAddress || EndAddress  ||  ReadMask   ||  WriteMask   ||Size of region||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||          ||              ||            ||              ||             ||             ||              ||              ||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");

    for(falconId = 0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        dprintf("||  %s  ||       %d      ||     -      ||      -       ||       -     ||      -      ||        -     ||      -       ||\n", lsFalc[falconId].name, lsFalc[falconId].size);
        for(index = 0; index < lsFalc[falconId].size; index++)
        {
            valCfga   = GPU_REG_RD32(lsFalc[falconId].regBaseCfga + (index*8));
            valCfgb   = GPU_REG_RD32(lsFalc[falconId].regBaseCfgb + (index*8));

            startAddr = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ADDR_LO, valCfga);
            readMask  = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ALLOW_READ, valCfga);

            endAddr   = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ADDR_HI, valCfgb);
            writeMask = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ALLOW_WRITE, valCfgb);

            //
            // Have used LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT here
            // as there is no generic define for WPR_ADDR_LO alignment,
            // and use of LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT is not to
            // be done anymore. (Same in case of WPR_ADDR_HI alignment define)
            //
            // TODO: Once there is a generic define added to the manual, 
            //       replace the below alignments with the new ones.
            //       (i.e LW_PFB_PRI_MMU_WPR_ADDR_LO_ALIGNMENT)
            //
            startAddr = startAddr << LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT;
            endAddr   = endAddr   << LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT;
            endAddr  += LW_MWPR_DEFAULT_ALIGNMENT - 1;

            _printFlcnWprInfo( index, startAddr, endAddr, readMask, writeMask);
        }
    }
    return LW_OK;
}

//-----------------------------------------------------------------------------
// acrVerifyMultipleWprStatus_TU10X - Verify Multiple Wpr regions configuration
//-----------------------------------------------------------------------------
LW_STATUS acrVerifyMultipleWprStatus_TU10X( LwU32 indexGpu)
{
    LwU32      falconId;
    LwU64      startAddr;
    LwU64      wprStartAddr[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1-1];
    LwU64      endAddr;
    LwU64      wprEndAddr[LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1-1];
    LwU32      readMask;
    LwU32      writeMask;
    LwU32      index;
    LwU32      valCfga;
    LwU32      valCfgb;
    LwU32      regionId;
    LwU32      cfgPLM;
    LwU32      readVal;
    LwU32      numSubWpr;
    LwU32      ScratchCfga;
    LwU32      ScratchCfgb;
    LwU32      ScratchPlm;

    dprintf("This command is not supporting PMU & SEC2 for now\n");

    dprintf("Verify mWPR configuration for all other supported falcons\n");
    // Verify Big WPRs configuration
    for (regionId=1; regionId <= (LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1-1); regionId++)
    {
        wprStartAddr[regionId-1] = REF_VAL(LW_PFB_PRI_MMU_WPR1_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ADDR_LO(regionId)));
        wprEndAddr[regionId-1]   = REF_VAL(LW_PFB_PRI_MMU_WPR1_ADDR_HI_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ADDR_HI(regionId)));

        readMask = DRF_IDX_VAL(_PFB, _PRI_MMU_WPR_ALLOW_READ, _WPR,
                regionId, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_READ));
        if(readMask != LW_WPR_EXPECTED_READ_MASK)
        {
            dprintf("WPR region %d AllowRead configuration error. Configured AllowRead mask is %x, expected is %x (L3 read)\n", regionId, readMask, LW_WPR_EXPECTED_READ_MASK);
        }
        else
        {
            dprintf("Big WPR region %d AllowRead configuration is correct\n", regionId);
        }

        writeMask = DRF_IDX_VAL(_PFB, _PRI_MMU_WPR_ALLOW_WRITE, _WPR,
                regionId, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_WRITE));
        if(writeMask != LW_WPR_EXPECTED_READ_MASK)
        {
            dprintf("WPR region %d AllowWrite configuration error. Configured AllowWrite mask is %x, expected is %x (L3 write)\n", regionId, writeMask, LW_WPR_EXPECTED_WRITE_MASK);
        }
        else
        {
            dprintf("Big WPR region %d AllowWrite configuration is correct\n", regionId);
        }
    }

    dprintf("SubWPR configuration status according to falcons\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("|| FalconId ||No. of Regions||Region Index|| Test Result                                                                ||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||          ||              ||            ||                                                                            ||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------\n");

    for(falconId = 0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        if ((falconId == LSF_FALCON_ID_PMU) || (falconId == LSF_FALCON_ID_SEC2))
        {
            continue;
        }

        numSubWpr = 0;

        if(lsFalc[falconId].size == 0)
        {
            // Skip for Minion
            dprintf("||  %s  ||       %d      ||     -      || N/A                                                                \n", lsFalc[falconId].name, lsFalc[falconId].size);
            continue;
        }

        cfgPLM    = GPU_REG_RD32(lsFalc[falconId].regCfgPLM);
        writeMask = REF_VAL(LW_PFB_PRI_MMU_SEC_PRIV_LEVEL_MASK_WRITE_PROTECTION, cfgPLM);

        if(writeMask != LW_PFB_PRI_MMU_SEC_PRIV_LEVEL_MASK_WRITE_PROTECTION_ONLY_LEVEL3_ENABLED)
        {
            dprintf("||  %s  ||       %d      ||     -      || CFG*_PLM write mask configuration error: Expected =  %01x, configured = %01x\n",
                    lsFalc[falconId].name, lsFalc[falconId].size, LW_PFB_PRI_MMU_SEC_PRIV_LEVEL_MASK_WRITE_PROTECTION_ONLY_LEVEL3_ENABLED, writeMask);
        }
        else
        {
            dprintf("||  %s  ||       %d      ||     -      ||      -                                                                      \n", lsFalc[falconId].name, lsFalc[falconId].size);
        }

        for(index = 0; index < lsFalc[falconId].size; index++)
        {
            valCfga   = GPU_REG_RD32(lsFalc[falconId].regBaseCfga + (index*8));
            valCfgb   = GPU_REG_RD32(lsFalc[falconId].regBaseCfgb + (index*8));

            startAddr = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ADDR_LO, valCfga);
            readMask  = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ALLOW_READ, valCfga);

            endAddr   = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ADDR_HI, valCfgb);
            writeMask = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ALLOW_WRITE, valCfgb);

            //
            // Have used LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT here
            // as there is no generic define for WPR_ADDR_LO alignment,
            // and use of LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT is not to
            // be done anymore. (Same in case of WPR_ADDR_HI alignment define)
            //
            // TODO: Once there is a generic define added to the manual, 
            //       replace the below alignments with the new ones.
            //       (i.e LW_PFB_PRI_MMU_WPR_ADDR_LO_ALIGNMENT)
            //
            startAddr = startAddr << LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT;
            endAddr   = endAddr   << LW_PFB_PRI_MMU_WPR1_ADDR_HI_ALIGNMENT;

            // Validate MMU config against secure scratch config
            if(LW_OK == _acrGetSelwreScratchAddrForSubWpr(falconId, index, &ScratchCfga, &ScratchCfgb, &ScratchPlm))
            {
                // SubWPR ID found, compare config
                if((valCfga != ScratchCfga) || (valCfgb != ScratchCfgb))
                {
                    dprintf("||          ||              ||     %d      || MMU config does not match Scratch config                       ||\n", index);
                    continue;
                }
            }

            // EndAddress should be greater than StartAddress
            if(endAddr < startAddr)
            {
                dprintf("||          ||              ||     %d      || EndAddress "LwU40_FMT" is less than StartAddress "LwU40_FMT"   ||\n", index, endAddr, startAddr);
                continue;
            }
            // If region size is non-zero, check further configuration
            else if(endAddr > startAddr)
            {
                // AllowRead configuration erorr
                if(!(readMask & LW_SUBWPR_EXPECTED_READ_MASK))
                {
                    dprintf("||          ||              ||     %d      || AllowRead configuration eror: Configured is %01x, Expected is >= L2 read ||\n", index, readMask);
                    continue;
                }
                else if(!(writeMask & LW_SUBWPR_EXPECTED_WRITE_MASK)) //AllowWrite configuration erorr
                {
                    dprintf("||          ||              ||     %d      || AllowWrite configuration eror: Configured is %01x, Expected is >= L2 write||\n", index, writeMask);
                    continue;
                }
                else // AllowRead/Write configuration correct
                {
                    // Try to read from SubWPR, result should be 0xBAD0----
                    {
                        pFb[indexGpu].fbRead(startAddr, &readVal, sizeof(LwU32));

                        if(0xBAD0 != (readVal >> 16) )
                        {
                            dprintf("||          ||              ||     %d      || Read error                                                     ||\n", index);
                            continue;
                        }
                    }
                }

                // SubWPR lies in only one Big WPR?
                if(((wprStartAddr[0] <= startAddr) && (wprEndAddr[0] >= endAddr)) ||
                        ((wprStartAddr[1] <= startAddr) && (wprEndAddr[1] >= endAddr)))
                {
                    dprintf("||          ||              ||     %d      || OK                                                             ||\n", index);
                    numSubWpr++;
                }
                else
                {
                    dprintf("||          ||              ||     %d      || SubWPR lies outside Big WPRs                                   ||\n", index);
                }
            }
            else
            {
                dprintf("||          ||              ||     %d      || SubWPR size zero (region not configured)                       ||\n", index);
            }
        }

        // Check constraint- Falcon which is in LS mode must have at least 2 SubWPRs configured: one for code and other for data
        if(lsFalc[falconId].bFalconEnabled == LW_TRUE)
        {
            // Falcon in LS mode?
            if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, GPU_REG_RD32(lsFalc[falconId].regBase + LW_PFALCON_FALCON_SCTL)) == 1)
            {
                if(numSubWpr < 2)
                {
                    dprintf("||  %s  ||       -      ||     -      || Falcon in LS must have at least 2 SubWPRs, configured = %d      ||\n", lsFalc[falconId].name, numSubWpr);
                }
            }
        }
    }

    return LW_OK;
}

//--------------------------------------------------------------------------------------------------------
// acrGetSharedWprStatus_TU10X - Verify Shared sub Wprs configuration
//--------------------------------------------------------------------------------------------------------
LW_STATUS acrGetSharedWprStatus_TU10X( LwU32 indexGpu)
{
    LwU32      size1;
    LwU32      size2;
    LW_STATUS  status;

    // LSF_SHARED_DATA_SUB_WPR_USE_CASE_ID_FRTS_VBIOS_TABLES: Shared between fbfalcon and PMU
    dprintf("Shared subWPR 1: FRTS\n");

    status = acrVerifySharedSubWpr(LSF_FALCON_ID_FBFALCON, FBFALCON_SUB_WPR_ID_2_FRTS_DATA_WPR1, ACR_SUB_WPR_RMASK_L2, ACR_SUB_WPR_WMASK_ALL_LEVELS_DISABLED, &size1);
    status |= acrVerifySharedSubWpr(LSF_FALCON_ID_PMU, PMU_SUB_WPR_ID_1_FRTS_DATA_WPR1, ACR_SUB_WPR_RMASK_L2, ACR_SUB_WPR_WMASK_ALL_LEVELS_DISABLED, &size2);

    // Both subWPR region sizes non-zero
    if(size1 && size2)
    {
        if(size1 != size2)
        {
            dprintf("Size mismatch: Shared subWPR FRTS between FbFalcon and PMU\n");
            status = LW_ERR_GENERIC;
        }
    }
    if(status == LW_OK)
    {
        dprintf("Shared subWPR 1 configuration - OK\n");
    }

    // LSF_SHARED_DATA_SUB_WPR_USE_CASE_ID_PLAYREADY_SHARED_DATA: Shared between SEC2 and LWDEC0
    dprintf("\nShared subWPR 2: Playready shared data\n");

    status = acrVerifySharedSubWpr(LSF_FALCON_ID_SEC2, SEC2_SUB_WPR_ID_1_PLAYREADY_SHARED_DATA_WPR1, ACR_SUB_WPR_RMASK_L3, ACR_SUB_WPR_WMASK_L3, &size1);
    status |= acrVerifySharedSubWpr(LSF_FALCON_ID_LWDEC, LWDEC0_SUB_WPR_ID_2_PLAYREADY_SHARED_DATA_WPR1, ACR_SUB_WPR_RMASK_L3, ACR_SUB_WPR_WMASK_L3, &size2);

    // Both subWPR region sizes non-zero
    if(size1 && size2)
    {
        if(size1 != size2)
        {
            dprintf("Size mismatch: Shared subWPR FRTS between FbFalcon and PMU\n");
            status = LW_ERR_GENERIC;
        }
    }
    if(status == LW_OK)
    {
        dprintf("Shared subWPR 2 configuration - OK\n");
    }

    return LW_OK;
}

LW_STATUS acrVerifySharedSubWpr(LwU32 falconId, LwU8 flcnSubWprId, LwU8 readMaskExp, LwU8 writeMaskExp, LwU32 *size)
{
    LwU32     startAddr;
    LwU32     endAddr;
    LwU32     readMask;
    LwU32     writeMask;
    LwU32     valCfga;
    LwU32     valCfgb;
    LW_STATUS status = LW_OK;

    valCfga   = GPU_REG_RD32(lsFalc[falconId].regBaseCfga + (flcnSubWprId*8));
    valCfgb   = GPU_REG_RD32(lsFalc[falconId].regBaseCfgb + (flcnSubWprId*8));

    startAddr = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ADDR_LO, valCfga);
    readMask  = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGA_ALLOW_READ, valCfga);

    endAddr   = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ADDR_HI, valCfgb);
    writeMask = REF_VAL(LW_PFB_PRI_MMU_FALCON_PMU_CFGB_ALLOW_WRITE, valCfgb);

    if(endAddr <= startAddr)
    {
        dprintf("Address configuration error for %s\n", lsFalc[falconId].name);
        status = LW_ERR_GENERIC;
    }
    else
    {
        *size = endAddr - startAddr;
    }
    if(readMaskExp != readMask)
    {
        dprintf("AllowRead configuration error for %s. Expected = %01x, configured = %01x\n", lsFalc[falconId].name, readMaskExp, readMask);
        status = LW_ERR_GENERIC;
    }
    if(writeMaskExp != writeMask)
    {
        dprintf("AllowWrite configuration error for %s. Expected = %01x, configured = %01x\n", lsFalc[falconId].name, writeMaskExp, writeMask);
        status = LW_ERR_GENERIC;
    }

    return status;
}

//------------------------------------------------------------
// _printFlcnWprInfo - Prints the info of mWPR regions in ACR
//------------------------------------------------------------
static LW_STATUS _printFlcnWprInfo(LwU32 regionId, LwU64 startAddr, LwU64 endAddr, LwU32 readMask, LwU32 writeMask)
{
    if (startAddr < endAddr)
    {
        dprintf("||          ||              ||     %d      || "  LwU40_FMT  " || "  LwU40_FMT  "||    %01x        ||      %01x       || " LwU40_FMT " ||\n", regionId, startAddr, endAddr, readMask, writeMask, (endAddr - startAddr + 1));
    }
    else
    {
        dprintf("||          ||              ||     %d      ||   Disabled   ||  Disabled   ||  Disabled   ||   Disabled   ||   Disabled   ||\n", regionId);
    }
    return LW_OK;
}


static LW_STATUS _acrGetSelwreScratchAddrForSubWpr(LwU32 falconId, LwU32 flcnSubWprId, LwU32 *pScratchCfga, LwU32 *pScratchCfgb, LwU32 *pScratchPlm)
{
    LW_STATUS status = LW_OK;

    switch (falconId)
    {
        /*case LSF_FALCON_ID_PMU:
            switch(flcnSubWprId)
            {
                case PMU_SUB_WPR_ID_0_ACRLIB_FULL_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case PMU_SUB_WPR_ID_1_FRTS_DATA_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_PMU_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;*/
        case LSF_FALCON_ID_FECS:
            switch(flcnSubWprId)
            {
                case FECS_SUB_WPR_ID_0_UCODE_CODE_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case FECS_SUB_WPR_ID_1_UCODE_DATA_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_FECS_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;
        /*case LSF_FALCON_ID_SEC2:
            switch(flcnSubWprId)
            {
                case SEC2_SUB_WPR_ID_0_ACRLIB_FULL_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case SEC2_SUB_WPR_ID_1_PLAYREADY_SHARED_DATA_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_SEC2_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;*/
        case LSF_FALCON_ID_LWDEC:
            switch(flcnSubWprId)
            {
                case LWDEC0_SUB_WPR_ID_0_UCODE_CODE_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case LWDEC0_SUB_WPR_ID_1_UCODE_DATA_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                case LWDEC0_SUB_WPR_ID_2_PLAYREADY_SHARED_DATA_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_2_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_2_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_LWDEC0_SUB_WPR_ID_2_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;
        case LSF_FALCON_ID_FBFALCON:
            switch(flcnSubWprId)
            {
                case FBFALCON_SUB_WPR_ID_0_UCODE_CODE_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case FBFALCON_SUB_WPR_ID_1_UCODE_DATA_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                case FBFALCON_SUB_WPR_ID_2_FRTS_DATA_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_2_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_2_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_FBFALCON_SUB_WPR_ID_2_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;
        case LSF_FALCON_ID_GSPLITE:
            switch(flcnSubWprId)
            {
                case GSP_SUB_WPR_ID_0_UCODE_CODE_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_0_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_0_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_0_PRIV_LEVEL_MASK;
                    break;
                case GSP_SUB_WPR_ID_1_UCODE_DATA_SECTION_WPR1:
                    *pScratchCfga = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_1_CFGA;
                    *pScratchCfgb = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_1_CFGB;
                    *pScratchPlm  = LW_PGC6_SELWRE_SCRATCH_GSP_SUB_WPR_ID_1_PRIV_LEVEL_MASK;
                    break;
                default:
                    status = LW_ERR_GENERIC;
                    break;
            }
            break;
        default:
            status = LW_ERR_GENERIC;
            break;
    }
    return status;
}

//------------------------------------------------------------
// acrGetRegionInfo_TU10X - Get ACR region info
//------------------------------------------------------------
LW_STATUS acrGetRegionInfo_TU10X( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32        cmd;
    LwU32        regionId;

    dprintf("Number of regions %d\n", LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1 - 1);

    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||  regionId  ||  StartAddress  ||  EndAddress    ||  Size          ||  ReadMask             ||  WriteMask           ||\n");
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||            ||                ||                ||                ||   P3   P2   P1   P0   ||   P3   P2   P1   P0  ||\n");
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");

    for (regionId = 1; regionId <= (LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1 - 1); regionId++)
    {

        if ((regionId - 1) == ACR_WPR1_REGION_IDX)
        {
            // Read start address for WPR1
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR1_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR1_ADDR_LO));
            startAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT;

            // Read end address for WPR1
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR1_ADDR_HI_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR1_ADDR_HI));
            endAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR1_ADDR_HI_ALIGNMENT;
        }
        else if ((regionId - 1) == ACR_WPR2_REGION_IDX)
        {
            // Read start address for WPR2
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR2_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR2_ADDR_LO));
            startAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR2_ADDR_LO_ALIGNMENT;

            // Read end address for WPR2
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR2_ADDR_HI_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR2_ADDR_HI));
            endAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR2_ADDR_HI_ALIGNMENT;
        }
        else
        {
            dprintf("The Region ID: %d is invalid.\n", regionId);
            continue;
        }

        // Check for a valid ACR region
        if (startAddr[regionId] <= endAddr[regionId])
        {
            // End address always point to start of the last aligned address
            endAddr[regionId] += (LW_ACR_DEFAULT_ALIGNMENT - 1);
        }
        else
        {
            // Not a valid ACR region. Set start and end as zero
            startAddr[regionId] = 0;
            endAddr[regionId]   = 0;
        }

        regionSize[regionId] = endAddr[regionId] - startAddr[regionId];

        // Read ReadMask
        cmd = GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_READ);
        readMask[regionId] = DRF_IDX_VAL(_PFB, _PRI_MMU_WPR, _ALLOW_READ_WPR, regionId, cmd);

        // Read WriteMask
        cmd = GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_WRITE);
        writeMask[regionId] = DRF_IDX_VAL(_PFB, _PRI_MMU_WPR, _ALLOW_WRITE_WPR, regionId, cmd);

        dprintf("||  %d         ||  " LwU40_FMT "  ||  " LwU40_FMT "  ||  " LwU40_FMT "  ||    %d    %d    %d    %d   ||    %d    %d    %d    %d  ||\n",
                    regionId, startAddr[regionId], endAddr[regionId], regionSize[regionId],
                    (readMask[regionId]>>3)&1, (readMask[regionId]>>2)&1, (readMask[regionId]>>1)&1, readMask[regionId]&1,
                    (writeMask[regionId]>>3)&1, (writeMask[regionId]>>2)&1, (writeMask[regionId]>>1)&1, writeMask[regionId]&1);
    }
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    return status;
}

//-------------------------------------------------------------------------------------------------------------
// acrRegionStatus_TU10X - L0 Sanity testing - Verify if write is possible through NS client
//-------------------------------------------------------------------------------------------------------------
LW_STATUS acrRegionStatus_TU10X( LwU32 indexGpu )
{
    LW_STATUS  status   = LW_OK;
    LwU32      value;
    LwU32      oldVal;
    LwU32      newVal;
    LwU32      cmd;
    LwU32      regionId;
    BOOL       modified = FALSE;

    dprintf("----------------------------------\n");
    dprintf("||  regionId  ||  Status        ||\n");
    dprintf("----------------------------------\n");

    for (regionId = 1; regionId <= (LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR__SIZE_1 - 1); regionId++)
    {
        // Read start address
        if ((regionId - 1) == ACR_WPR1_REGION_IDX)
        {
            // Read start address for WPR1
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR1_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR1_ADDR_LO));
            startAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR1_ADDR_LO_ALIGNMENT;
        }
        else if ((regionId - 1) == ACR_WPR2_REGION_IDX)
        {
            // Read start address for WPR2
            cmd   = REF_VAL(LW_PFB_PRI_MMU_WPR2_ADDR_LO_VAL, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR2_ADDR_LO));
            startAddr[regionId] = (LwU64)cmd << LW_PFB_PRI_MMU_WPR2_ADDR_LO_ALIGNMENT;
        }
        else
        {
            dprintf("The Region ID: %d is invalid.\n", regionId);
            continue;
        }

        // Read bit0 ReadMask
        readMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR, _ALLOW_READ_WPR_SELWRE,
                                                regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_READ));

        // Read bit0 WriteMask
        writeMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR, _ALLOW_WRITE_WPR_SELWRE,
                                                regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_ALLOW_WRITE));

        dprintf("||  %d         ||  ", regionId);

        if (readMask[regionId] != LW_PFB_PRI_MMU_WPR_ALLOW_READ_WPR1_SELWRE0_DEFAULT)
        {
            dprintf("Failed        ||\n");
            continue;
        }

        pFb[indexGpu].fbRead(startAddr[regionId], &oldVal, sizeof(LwU32));

        // Generating a new value by toggling bits;
        value = oldVal ^ 0xffffffff;

        pFb[indexGpu].fbWrite(startAddr[regionId], &value, sizeof(LwU32));
        pFb[indexGpu].fbRead(startAddr[regionId], &newVal, sizeof(LwU32));

        if (writeMask[regionId] != LW_PFB_PRI_MMU_WPR_ALLOW_WRITE_WPR1_SELWRE0_DEFAULT)
        {
            if (newVal == value)
            {
                dprintf("Success       ||\n");
                modified = TRUE;
            }
            else
            {
                dprintf("Failed        ||\n");
            }
        }
        else
        {
            if (newVal == value)
            {
                dprintf("Failed        ||\n");
                modified = TRUE;
            }
            else
            {
                dprintf("Success       ||\n");
            }
        }

        if (modified == TRUE)
        {
            pFb[indexGpu].fbWrite(startAddr[regionId], &oldVal, sizeof(LwU32));
        }
    }
    dprintf("----------------------------------\n");
    return status;
}


