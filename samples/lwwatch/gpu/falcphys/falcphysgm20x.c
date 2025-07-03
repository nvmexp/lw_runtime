/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "falcphys.h"
#include "os.h"
#include "rmlsfm.h"
#include "disp.h"
#include "pmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "clk.h"
#include "smbpbi.h"
#include "lwtypes.h"
#include "acr.h"

#include "g_falcphys_private.h"          // (rmconfig) hal/obj setup
#include "g_acr_hal.h"
#include "g_fb_hal.h"

#include "maxwell/gm200/dev_fb.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "maxwell/gm200/dev_fbif_v4.h"
#include "maxwell/gm200/dev_fuse.h"
#include "maxwell/gm200/dev_graphics_nobundle.h"
#include "maxwell/gm200/dev_sec_pri.h"
#include "maxwell/gm200/dev_master.h"
#include "maxwell/gm200/dev_lwenc_pri_sw.h"
#include "maxwell/gm200/dev_pwr_pri.h"
#include "maxwell/gm200/dev_hdafalcon_pri.h"

FALCONPROP  falc[10];
LwU32       buffer1[64];
LwU32       buffer2[64];
LwU32       buffer3[64];
FALCMMUPHYS falcMmuStruct;

void falcphysGetMmuPhysRegConfig_GM20X (void *pFalcVoid)
{
    FALCMMUPHYS *pFalcMmu = (FALCMMUPHYS *) pFalcVoid;
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
//-----------------------------------------------------
// falcphysIsSupported_GM20X
//-----------------------------------------------------
BOOL falcphysIsSupported_GM20X( LwU32 indexGpu )
{
    LwU32 reg;

    // Skip IsSupported check in case of fmodel
    if (pAcr[indexGpu].acrIsFmodel())
    {
        return TRUE;
    }

    // Check PRIV_SEC fuse, if fuse is not blown then PRIV sec feature in not present
    reg = GPU_REG_RD32(LW_FUSE_OPT_PRIV_SEC_EN);
    if (DRF_VAL(_FUSE, _OPT_PRIV_SEC_EN, _DATA, reg) == LW_FUSE_OPT_PRIV_SEC_EN_DATA_NO)
    {
        dprintf("Need PRIV_SEC fuse blown for priv security\n");
        return FALSE;
    }

    // Check WPR fuse, if fuse is not blown, then wpr feature is not present
    reg = GPU_REG_RD32(LW_FUSE_OPT_WPR_ENABLED);
    if (DRF_VAL(_FUSE, _OPT_WPR_ENABLED_, DATA, reg) == LW_FUSE_OPT_WPR_ENABLED_DATA_NO)
    {
        dprintf("Need WPR fuse blown for wpr feature\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------------------------------------------
// falcphysDmaAccessCheck - Verifies falcon physical DMA access restrictions
//-----------------------------------------------------------------------------------------
LW_STATUS falcphysDmaAccessCheck_GM20X( LwU32 indexGpu )
{
    LW_STATUS  status;
    LwU32 reg, reg2, newVal;
    LwU32 i;
    LwU32 cmd;
    LwU32 fbVal;
    LwU32 imemVal;
    LwU32 dmemVal;
    LwU32 index   = 0;
    LwU32 falconId;
    LwU64 startAddr;

    reg = GPU_REG_RD32(LW_PFB_PRI_MMU_PRIV_LEVEL_MASK);
    if (DRF_VAL(_PFB, _PRI_MMU_PRIV_LEVEL_MASK_, WRITE_PROTECTION_LEVEL0, reg)
                == LW_PFB_PRI_MMU_PRIV_LEVEL_MASK_WRITE_PROTECTION_LEVEL0_ENABLE)
    {
        dprintf("FAIL - Priv security is not enabled, CPU has access\n"
                "Need a platform where priv security is enabled to run the test\n");
        return LW_ERR_GENERIC;
    }
    else
    {
        dprintf("Priv security is enabled\n");

        pFalcphys[indexGpu].falcphysGetMmuPhysRegConfig((void *)&falcMmuStruct);

        reg2 = GPU_REG_RD32(falcMmuStruct.regBase);
        newVal = reg2 ^ 0xffffffff;
        GPU_REG_WR32(falcMmuStruct.regBase, newVal);

        if (GPU_REG_RD32(falcMmuStruct.regBase) == newVal)
        {
            dprintf("FAIL - CPU still has access\n");
            return LW_ERR_GENERIC;
        }
        else
        {
            dprintf("PASS - CPU not able to change contents of LW_PFB_PRI_MMU_PHYS_SELWRE, "
                    "current value: 0X%08X\n", reg2);
        }
    }

    // Let us take a random FB address for verification, say 0x1000
    startAddr = 0x1000;

    // preparing buffer
    for(i = 0; i < TARGET_BUFFER_SIZE/4; i++)
    {
        buffer1[i] = i;
        buffer2[i] = i + (TARGET_BUFFER_SIZE/4);
        buffer3[i] = i + (TARGET_BUFFER_SIZE/2);
    }

    dprintf("--------------------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("|| Falcon  |   Is    | MMU_PHYS_SELWRE |  Is Manual  | Has NSMODE physical    |   Is    |     Read Results    |    Write Results    ||\n");
    dprintf("||         | Present |    reg value    |   Default?  | DMA access restriction | NSMODE  | FB->IMEM | FB->DMEM | IMEM->FB | DMEM->FB ||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------------------\n");

    //
    // verify for each falcon
    //
    for (falconId = 0; falconId < TOTAL_FALCONS; falconId++)
    {
        status = LW_OK;

        switch(falconId)
        {
            case FECS_FALCON_ID:
                    falc[falconId].name          = "FECS   ";
                    falc[falconId].regBase       = LW_PGRAPH_PRI_FECS_FALCON_IRQSSET;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _PGRAPH, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, FECS, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.fecsInit;
                    break;
            case PMU_FALCON_ID:
                    falc[falconId].name          = "PMU    ";
                    falc[falconId].regBase       = LW_FALCON_PWR_BASE;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _PWR, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, PMU, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.pmuInit;
                    break;
            case SEC_FALCON_ID:
                    falc[falconId].name          = "SEC    ";
                    falc[falconId].regBase       = LW_PSEC_FALCON_IRQSSET;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _SEC, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, SEC, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.secInit;
                    break;
            case DPU_FALCON_ID:
                    falc[falconId].name          = "DPU    ";
                    falc[falconId].regBase       = LW_FALCON_DISP_BASE;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _PDISP, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, DFALCON, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.dfalconInit;
                    break;
            case HDA_FALCON_ID:                              // There is no PMC bit defined for HDA Falcon
                    falc[falconId].name          = "HDA    ";
                    falc[falconId].regBase       = LW_FALCON_HDA_BASE;
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, AFALCON, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.afalconInit;
                    break;
            case LWDEC_FALCON_ID:
                    falc[falconId].name          = "LWDEC  ";
                    falc[falconId].regBase       = LW_FALCON_LWDEC_BASE;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _LWDEC, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, LWDEC, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.lwdecInit;
                    break;
            case LWENC0_FALCON_ID:
                    falc[falconId].name          = "LWENC0 ";
                    falc[falconId].regBase       = LW_FALCON_LWENC0_BASE;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _LWENC0, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, LWENC, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.lwencInit;
                    break;
            case LWENC1_FALCON_ID:
                    falc[falconId].name          = "LWENC1 ";
                    falc[falconId].regBase       = LW_FALCON_LWENC1_BASE;
                    falc[falconId].pmcMask       = DRF_DEF(_PMC, _ENABLE, _LWENC1, _ENABLED);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, LWENC, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.lwencInit;
                    break;
            case LWENC2_FALCON_ID:
                    falc[falconId].name          = "LWENC2 ";
                    pFalcphys[indexGpu].falcphysGetLwenc2FalcPhysProp(falc);
                    falc[falconId].selwreFalc    = DRF_VAL(_PFB, _PRI_MMU_PHYS_SELWRE_, LWENC, reg2);
                    falc[falconId].selwreInitVal = falcMmuStruct.lwencInit;
                    break;
            default:
                    dprintf("Invalid falconId\n");
                    return LW_ERR_GENERIC;
        }

        dprintf("|| %s ", falc[falconId].name);

        if (pFalcphys[indexGpu].falcphysCheckEngineIsPresent(falconId))
        {
            dprintf("| Yes     | 0x%08X      ", falc[falconId].selwreFalc);

            if (falc[falconId].selwreFalc == falc[falconId].selwreInitVal)
            {
                dprintf("| Yes         ");

                if (falc[falconId].selwreFalc == LW_PFB_PRI_MMU_PHYS_SELWRE_FALCON_NO)
                {
                    dprintf("| No                     | N/A                                                 ||\n");
                    status = LW_ERR_GENERIC;
                }
                else
                {
                    dprintf("| Yes                    ");
                }
            }
            else
            {
                dprintf("| WARNING: No ");

                if (falc[falconId].selwreFalc == LW_PFB_PRI_MMU_PHYS_SELWRE_FALCON_NO)
                {
                    dprintf("| No-FAIL                | N/A                                                 ||\n");
                    status = LW_ERR_GENERIC;
                }
                else
                {
                    dprintf("| Yes                    ");
                }
            }

            if(status == LW_OK)
            {
                //
                // Reset falcons by setting PMC_ENABLE bits, but HDA needs to be disabled using regkey
                //
                if (falconId == HDA_FALCON_ID)
                {
                    cmd = 0;
                    cmd = FLD_SET_DRF_NUM(_PHDAFALCON, _FALCON_CPUCTL, _SRESET, 1, cmd);
                    cmd = FLD_SET_DRF_NUM(_PHDAFALCON, _FALCON_CPUCTL, _HRESET, 1, cmd);
                    GPU_REG_WR32(LW_PHDAFALCON_FALCON_CPUCTL, cmd);

                    if (GPU_REG_RD32(LW_PHDAFALCON_FALCON_CPUCTL) != DRF_SHIFTMASK(LW_PHDAFALCON_FALCON_CPUCTL_HALTED))
                    {
                        dprintf("| FAIL - Could not reset HDA Falcon, use regkey 'RmEnableHda' "
                                "by adding '-enable_hda 0x0' in command line to disable HDA falcon ||\n");
                        status = LW_ERR_GENERIC;
                    }
                }
                else if (falc[falconId].pmcMask != 0)
                {
                    reg = GPU_REG_RD32(LW_PMC_ENABLE);
                    cmd = reg & ~falc[falconId].pmcMask;
                    GPU_REG_WR32(LW_PMC_ENABLE,cmd);

                    // Do an extra read after writing LW_PMC_ENABLE
                    GPU_REG_RD32(LW_PMC_ENABLE);

                    cmd = reg | falc[falconId].pmcMask;
                    GPU_REG_WR32(LW_PMC_ENABLE,cmd);

                    // Do an extra read after writing LW_PMC_ENABLE
                    GPU_REG_RD32(LW_PMC_ENABLE);
                }
            }

            if (status == LW_OK)
            {
                // Check if Falcon is is NSMODE after reset
                reg = GPU_REG_RD32(falc[falconId].regBase+LW_PFALCON_FALCON_SCTL);

                if ((DRF_VAL(_PFALCON_FALCON, _SCTL_, HSMODE, reg) == LW_PFALCON_FALCON_SCTL_HSMODE_TRUE) ||
                    (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, reg) == LW_PFALCON_FALCON_SCTL_LSMODE_TRUE))
                {
                    dprintf("| No-FAIL | N/A                                       ||\n");
                    status = LW_ERR_GENERIC;
                }
                else
                {
                    dprintf("| Yes     ");
                }

                if (status == LW_OK)
                {
                    if(falconId == FECS_FALCON_ID)  // FECS falcon uses ARBITER instead of FBIF to access FB
                    {
                        reg = GPU_REG_RD32(LW_PGRAPH_PRI_FECS_ARB_WPR);
                        reg = FLD_SET_DRF(_PGRAPH, _PRI_FECS_ARB_WPR, _CMD_OVERRIDE_PHYSICAL_WRITES, _ALLOWED, reg);
                        GPU_REG_WR32(LW_PGRAPH_PRI_FECS_ARB_WPR, reg);

                        reg = GPU_REG_RD32(LW_PGRAPH_PRI_FECS_ARB_CMD_OVERRIDE);
                        reg = FLD_SET_DRF(_PGRAPH, _PRI_FECS_ARB_CMD_OVERRIDE, _CMD, _PHYS_VID_MEM, reg);
                        reg = FLD_SET_DRF(_PGRAPH, _PRI_FECS_ARB_CMD_OVERRIDE, _ENABLE, _ON, reg);
                        GPU_REG_WR32(LW_PGRAPH_PRI_FECS_ARB_CMD_OVERRIDE, reg);
                    }
                    else
                    {
                        // Set TRANSCFG reg to have physical access to FB
                        if(falconId == PMU_FALCON_ID)
                        {
                            reg = GPU_REG_RD32(LW_PPWR_FBIF_TRANSCFG(CTX_DMA_ID));
                        }
                        else if(falconId == LWENC0_FALCON_ID || falconId == LWENC1_FALCON_ID || falconId == LWENC2_FALCON_ID)
                        {
                            reg = GPU_REG_RD32(LW_PLWENC_FBIF_TRANSCFG((falconId - 6), CTX_DMA_ID));
                        }
                        else
                        {
                            reg = GPU_REG_RD32((falc[falconId].regBase + LW_FALCON_FBIF_TRANSCFG(CTX_DMA_ID)));
                        }
                        cmd = FLD_SET_DRF(_PFALCON, _FBIF_TRANSCFG,   _TARGET, _LOCAL_FB, reg);
                        cmd = FLD_SET_DRF(_PFALCON, _FBIF_TRANSCFG, _MEM_TYPE, _PHYSICAL, cmd);

                        if(falconId == PMU_FALCON_ID)
                        {
                            GPU_REG_WR32(LW_PPWR_FBIF_TRANSCFG(CTX_DMA_ID), cmd);
                        }
                        else if(falconId == LWENC0_FALCON_ID || falconId == LWENC1_FALCON_ID || falconId == LWENC2_FALCON_ID)
                        {
                            GPU_REG_WR32((LW_PLWENC_FBIF_TRANSCFG((falconId - 6), CTX_DMA_ID)), cmd);
                        }
                        else
                        {
                            GPU_REG_WR32((falc[falconId].regBase + LW_FALCON_FBIF_TRANSCFG(CTX_DMA_ID)), cmd);
                        }
                    }

                    // populating target buffer on FB
                    for(i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
                    {
                        pFb[indexGpu].fbWrite((startAddr + (i*4)), &buffer1[i], sizeof(LwU32));
                    }

                    // verifying if populated
                    for(i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
                    {
                        pFb[indexGpu].fbRead((startAddr + (i*4)), &fbVal, sizeof(LwU32));
                        if(fbVal != buffer1[i])
                        {
                            dprintf("couldnt populate target buffer on FB");
                            return LW_ERR_GENERIC;
                        }
                    }

                    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_IMEMC(index)),
                                  DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_AINCW));
                    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMEMC(index)),
                                  DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_AINCW));

                    // populate IMEM/DMEM with known values
                    for(i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
                    {
                        GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_IMEMD(index)), buffer2[i]);
                        GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMEMD(index)), buffer3[i]);
                    }

                    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_IMEMC(index)),
                                  DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_AINCR));
                    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMEMC(index)),
                                  DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_AINCR));

                    // verifying if IMEM/DMEM are populated
                    for(i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
                    {
                        imemVal = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_IMEMD(index));
                        dmemVal = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_DMEMD(index));

                        if (imemVal != buffer2[i])
                        {
                            dprintf("couldnt populate IMEM");
                            return LW_ERR_GENERIC;
                        }
                        if(dmemVal != buffer3[i])
                        {
                            dprintf("couldnt populate DMEM");
                            return LW_ERR_GENERIC;
                        }
                    }

                    //
                    // Reading from FB
                    //
                    // FB->IMEM
                    physicalDmaAccess(falconId, startAddr,  LW_TRUE, LW_FALSE);
                    if(verify(falconId, startAddr, LW_TRUE, LW_FALSE))
                    {
                        dprintf("| PASS     ");
                    }
                    else
                    {
                        dprintf("| FAIL     ");
                        status = LW_ERR_GENERIC;
                    }

                    // FB->DMEM
                    physicalDmaAccess(falconId, startAddr, LW_FALSE, LW_FALSE);
                    if(verify(falconId, startAddr, LW_FALSE, LW_FALSE))
                    {
                        dprintf("| PASS     ");
                    }
                    else
                    {
                        dprintf("| FAIL     ");
                        status = LW_ERR_GENERIC;
                    }

                    //
                    // Write to FB
                    //
                    if(status == LW_OK)
                    {
                        // IMEM->FB
                        physicalDmaAccess(falconId, startAddr, LW_TRUE, LW_TRUE);
                        if(verify(falconId, startAddr, LW_TRUE, LW_TRUE))
                        {
                            dprintf("| PASS     ");
                        }
                        else
                        {
                            dprintf("| FAIL     ");
                        }

                        // DMEM->FB
                        physicalDmaAccess(falconId, startAddr, LW_FALSE,  LW_TRUE);
                        if(verify(falconId, startAddr, LW_FALSE, LW_TRUE))
                        {
                            dprintf("| PASS     ||\n");
                        }
                        else
                        {
                            dprintf("| FAIL     ||\n");
                        }
                    }
                    else
                    {
                        dprintf("| N/A      | N/A      ||\n");
                    }
                }
            }
        }
        else
        {
            dprintf("| No      | N/A                                                                                                          ||\n");
        }
    }

    dprintf("-------------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("Warning - this test will corrupt FBMEM and so a system restart is required "
            "to resume normal system operation\n");
    return status;
}

void physicalDmaAccess(LwU32 falconId, LwU64 addr, LwBool imem, LwBool write)
{

    LwU32 data, dmaCmd;

    // Setup FB interface
    data = GPU_REG_RD32(falc[falconId].regBase + LW_FALCON_FBIF_CTL);
    data = FLD_SET_DRF(_PFALCON, _FBIF_CTL,            _ENABLE,  _TRUE, data);
    data = FLD_SET_DRF(_PFALCON, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW, data);
    GPU_REG_WR32((falc[falconId].regBase + LW_FALCON_FBIF_CTL), data);

    //
    // Program DMA registers
    // Write DMA base address
    //
    data = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_DMACTL);
    data = FLD_SET_DRF(_PFALCON, _FALCON_DMACTL, _REQUIRE_CTX, _FALSE, data);
    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMACTL), data);

    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFBASE),  LwU64_LO32(addr));

    if (falconId != HDA_FALCON_ID)
    {
        pFalcphys[indexGpu].falcphysProgramDmaBase1Reg(falc, falconId, addr);
    }

    // prepare DMA command
    {
        dmaCmd = 0;
        if (imem)
        {
            dmaCmd = FLD_SET_DRF(_PFALCON, _FALCON_DMATRFCMD, _IMEM, _TRUE, dmaCmd);
        }
        else
        {
            dmaCmd = FLD_SET_DRF(_PFALCON, _FALCON_DMATRFCMD, _IMEM, _FALSE, dmaCmd);
        }

        if (write) // flcn -> FB
        {
            dmaCmd = FLD_SET_DRF(_PFALCON, _FALCON_DMATRFCMD, _WRITE, _TRUE, dmaCmd);
        }
        else   // FB-> flcn
        {
            dmaCmd = FLD_SET_DRF(_PFALCON, _FALCON_DMATRFCMD, _WRITE, _FALSE, dmaCmd);
        }

        // Allow only 256B transfers
        dmaCmd = FLD_SET_DRF(_PFALCON, _FALCON_DMATRFCMD, _SIZE, _256B, dmaCmd);

        dmaCmd = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMATRFCMD, _CTXDMA, CTX_DMA_ID, dmaCmd);
    }

    data = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFMOFFS);
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMATRFMOFFS, _OFFS, 0, 0);
    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFMOFFS),data);

    data = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFFBOFFS);
    data = FLD_SET_DRF_NUM(_PFALCON, _FALCON_DMATRFFBOFFS, _OFFS, 0, 0);
    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFFBOFFS),data);

    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMATRFCMD),dmaCmd);
}

LwBool verify(LwU32 falconId, LwU64 addr, LwBool imem, LwBool write)
{
    LwU32 valFalc, valFb;
    LwU32 index = 0;
    LwU32 i;

    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_IMEMC(index)),
                  DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_AINCR));
    GPU_REG_WR32((falc[falconId].regBase + LW_PFALCON_FALCON_DMEMC(index)),
                  DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_AINCR));

    if (!write)     // verify Fb->Falc
    {
        if (imem)   // verify Fb->Imem
        {
            for (i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
            {
                valFalc = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_IMEMD(index));
                if (valFalc == buffer1[i])
                {
                    return LW_FALSE;
                }
            }
        }
        else        // verify Fb->Dmem
        {
            for (i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
            {
                valFalc = GPU_REG_RD32(falc[falconId].regBase + LW_PFALCON_FALCON_DMEMD(index));
                if (valFalc == buffer1[i])
                {
                    return LW_FALSE;
                }
            }
        }
    }
    else            // verify Flcn->Fb
    {
        if (imem)   // verify Imem->Fb
        {
            for (i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
            {
                pFb[indexGpu].fbRead((addr + (i*4)), &valFb, sizeof(LwU32));
                if (valFb == buffer2[i])
                {
                    return LW_FALSE;
                }
            }
        }
        else        // verify Dmem->Fb
        {
            for (i = 0; i < (TARGET_BUFFER_SIZE/4); i++)
            {
                pFb[indexGpu].fbRead((addr + (i*4)), &valFb, sizeof(LwU32));
                if (valFb == buffer3[i])
                {
                    return LW_FALSE;
                }
            }
        }
    }

    return LW_TRUE;
}

LwBool falcphysCheckEngineIsPresent_GM20X( LwU32 falconId )
{
    switch (falconId)
    {
        case FECS_FALCON_ID:
            if (GPU_REG_RD_DRF_IDX( _PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH)) == LW_PMC_ENABLE_DEVICE_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case PMU_FALCON_ID:
            return LW_TRUE;
            break;
        case SEC_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_SEC, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_SEC)) == LW_FUSE_STATUS_OPT_SEC_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case DPU_FALCON_ID:
        case HDA_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_DISPLAY, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_DISPLAY)) == LW_FUSE_STATUS_OPT_DISPLAY_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWDEC_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_LWDEC, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWDEC)) == LW_FUSE_STATUS_OPT_LWDEC_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC0_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 0, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC1_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 1, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC2_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 2, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        default:
            dprintf(" Invalid falconId ");
    }
    return LW_FALSE;
}
