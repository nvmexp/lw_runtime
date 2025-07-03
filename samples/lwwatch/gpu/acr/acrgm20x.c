/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2019 by LWPU Corporation.  All rights reserved.  All information
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

#include "g_acr_private.h"          // (rmconfig) hal/obj setup
#include "g_fb_private.h"

#include "dpu/v02_05/dev_disp_falcon.h"
#include "maxwell/gm200/dev_pwr_pri.h"
#include "maxwell/gm200/dev_lwenc_pri_sw.h"
#include "maxwell/gm200/dev_lwdec_pri.h"
#include "maxwell/gm200/dev_fb.h"
#include "maxwell/gm200/dev_fuse.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "maxwell/gm200/dev_graphics_nobundle.h"
#include "maxwell/gm200/dev_sec_pri.h"
#include "maxwell/gm200/dev_master.h"

#define LW_ACR_DEFAULT_ALIGNMENT 0x20000

LwU64   startAddr[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
LwU64   endAddr[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
LwU64   regionSize[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
LwU32   readMask[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
LwU32   writeMask[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];

LSFALCPROP lsFalc[LSF_FALCON_ID_END];

//-----------------------------------------------------
// acrIsSupported_GM20X
//-----------------------------------------------------
BOOL acrIsSupported_GM20X( LwU32 indexGpu )
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
    if(!FLD_TEST_DRF(_FUSE,_OPT_WPR_ENABLED,_DATA,_YES,fuse))
    {
        dprintf("ACR: WPR fuse is not blown\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------------------------------------------
// acrLsfStatus_GM20X - Prints PRIV level of falcons, if engine is enabled
//-----------------------------------------------------------------------------------------
LW_STATUS acrLsfStatus_GM20X(LwU32 indexGpu )
{
    LW_STATUS  status = LW_OK;
    LwU32 falconId;
    LwU32 reg;
    LwU32 reg1;
    LwU32 csbLevel;
    LwU32 extLevel;

    dprintf("-----------------------------------------------------------------------------------\n");
    dprintf("||  LsFalcon  ||  Engine    ||  Boot Mode  ||  Priv Level                        ||\n");
    dprintf("-----------------------------------------------------------------------------------\n");
    dprintf("||            ||            ||             ||  CSB interface  ||  EXT interface  ||\n");
    dprintf("-----------------------------------------------------------------------------------\n");

    for (falconId=0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        dprintf("||  %s    ", lsFalc[falconId].name);

        if (lsFalc[falconId].available == LW_FALSE)
        {
            dprintf("||  Not supported                                                  ||\n");
            continue;
        }

        if(LW_TRUE == lsFalc[falconId].bFalconEnabled)
        {
            dprintf("||  ENABLED   ");
            reg  = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_SCTL);
            reg1 = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_SCTL1);

            if (DRF_VAL(_PFALCON_FALCON, _SCTL_, HSMODE, reg) == 1)
            {
                dprintf("||  HSMODE     ");
                csbLevel = (3 & DRF_VAL(_PFALCON_FALCON, _SCTL1_, CSBLVL_MASK, reg1));
                extLevel = (3 & DRF_VAL(_PFALCON_FALCON, _SCTL1_, EXTLVL_MASK, reg1));
            }
            else if(DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, reg) == 1)
            {
                dprintf("||  LSMODE     ");
                csbLevel = DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE_LEVEL, reg) &
                                    DRF_VAL(_PFALCON_FALCON, _SCTL1_, CSBLVL_MASK, reg1);
                extLevel = DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE_LEVEL, reg) &
                                    DRF_VAL(_PFALCON_FALCON, _SCTL1_, EXTLVL_MASK, reg1);
            }
            else
            {
                dprintf("||  NSMODE     ");
                csbLevel = extLevel = 0;
            }
            dprintf("||  %d              ||  %d              ||\n", csbLevel, extLevel);
        }
        else
        {
            dprintf("||  DISABLED  ||  NA         ||                 ||                 ||\n");
        }
    }
    dprintf("-----------------------------------------------------------------------------------\n");
    return status;
}

//------------------------------------------------------------
// acrGetRegionInfo_GM20X - Get ACR region info
//------------------------------------------------------------
LW_STATUS acrGetRegionInfo_GM20X( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   cmd;
    LwU32   regionId;

    dprintf("Number of regions %d\n", LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1-1);

    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||  regionId  ||  StartAddress  ||  EndAddress    ||  Size          ||  ReadMask             ||  WriteMask           ||\n");
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||            ||                ||                ||                ||   P3   P2   P1   P0   ||   P3   P2   P1   P0  ||\n");
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");

    for (regionId=1; regionId <= (LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1-1); regionId++)
    {
        // Read start address
        cmd = FLD_SET_DRF_IDX(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _WPR_ADDR_LO, regionId, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        startAddr[regionId] = GPU_REG_RD_DRF(_PFB, _PRI_MMU_WPR_INFO, _DATA);
        startAddr[regionId] = startAddr[regionId] << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;

        // Read end address
        cmd = FLD_SET_DRF_IDX(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _WPR_ADDR_HI, regionId, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        endAddr[regionId] = GPU_REG_RD_DRF(_PFB, _PRI_MMU_WPR_INFO, _DATA);
        endAddr[regionId] = endAddr[regionId] << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;
        // End address always point to start of the last aligned address
        endAddr[regionId] += LW_ACR_DEFAULT_ALIGNMENT;

        regionSize[regionId] = endAddr[regionId] - startAddr[regionId];

        // Read ReadMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_READ, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        readMask[regionId] = GPU_REG_RD_DRF_IDX(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_READ_WPR, regionId);

        // Read WriteMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_WRITE, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        writeMask[regionId] = GPU_REG_RD_DRF_IDX(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_WRITE_WPR, regionId);

        dprintf("||  %d         ||  " LwU40_FMT "  ||  " LwU40_FMT "  ||  " LwU40_FMT "  ||    %d    %d    %d    %d   ||    %d    %d    %d    %d  ||\n",
                    regionId, startAddr[regionId], endAddr[regionId], regionSize[regionId],
                    (readMask[regionId]>>3)&1, (readMask[regionId]>>2)&1, (readMask[regionId]>>1)&1, readMask[regionId]&1,
                    (writeMask[regionId]>>3)&1, (writeMask[regionId]>>2)&1, (writeMask[regionId]>>1)&1, writeMask[regionId]&1);
    }
    dprintf("-----------------------------------------------------------------------------------------------------------------------\n");
    return status;
}

//-------------------------------------------------------------------------------------------------------------
// acrRegionStatus_GM20X - L0 Sanity testing - Verify if write is possible through NS client
//-------------------------------------------------------------------------------------------------------------
LW_STATUS acrRegionStatus_GM20X( LwU32 indexGpu )
{
    LW_STATUS    status   = LW_OK;
    LwU32   value;
    LwU32   oldVal;
    LwU32   newVal;
    LwU32   cmd;
    LwU32   regionId;
    BOOL    modified = FALSE;

    dprintf("----------------------------------\n");
    dprintf("||  regionId  ||  Status        ||\n");
    dprintf("----------------------------------\n");

    for (regionId=1; regionId <= (LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1-1); regionId++)
    {
        // Read start address
        cmd = FLD_SET_DRF_IDX(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _WPR_ADDR_LO, regionId, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        startAddr[regionId] = GPU_REG_RD_DRF(_PFB, _PRI_MMU_WPR_INFO, _DATA);
        startAddr[regionId] = startAddr[regionId] << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;

        // Read bit0 ReadMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_READ, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);

        readMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_READ_WPR_SELWRE,
                                                regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_INFO));
        // Read bit0 WriteMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_WRITE, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);

        writeMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_WRITE_WPR_SELWRE,
                                                 regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_INFO));

        dprintf("||  %d         ||  ", regionId);

        if(readMask[regionId] != LW_PFB_PRI_MMU_WPR_INFO_ALLOW_READ_WPR1_SELWRE0_DEFAULT)
        {
            dprintf("Failed        ||\n");
            continue;
        }

        pFb[indexGpu].fbRead(startAddr[regionId], &oldVal, sizeof(LwU32));

        // Generating a new value by toggling bits;
        value = oldVal ^ 0xffffffff;

        pFb[indexGpu].fbWrite(startAddr[regionId], &value, sizeof(LwU32));
        pFb[indexGpu].fbRead(startAddr[regionId], &newVal, sizeof(LwU32));

        if(writeMask[regionId] != LW_PFB_PRI_MMU_WPR_INFO_ALLOW_WRITE_WPR1_SELWRE0_DEFAULT)
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

        if(modified == TRUE)
        {
            pFb[indexGpu].fbWrite(startAddr[regionId], &oldVal, sizeof(LwU32));
        }
    }
    dprintf("----------------------------------\n");
    return status;
}

//-----------------------------------------------------
// acrDmemProtection_GM20X - Verify DMEM protection
//-----------------------------------------------------
LW_STATUS acrDmemProtection_GM20X( LwU32 indexGpu )
{
    LW_STATUS   status = LW_OK;
    LwU32  value;
    LwU32  oldVal;
    LwU32  newVal;
    LwU32  falconId;
    LwU32  reg;
    LwU32  reg1;
    LwU32  reg2;
    LwU32  range;
    LwU32  cmd;
    LwU32  rangeStart = 0;
    LwU32  rangeEnd   = 0;
    LwU32  offset;
    LwU32  index = 0;
    LwBool readProtection;
    LwBool writeProtection;
    LwU32  dmemSize;

    dprintf("---------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("|| Falcon | Is LSMode | Level-0 Write | Level-0 Read | Out of range                      |  RANGE 0                     | Range 1                      ||\n");
    dprintf("---------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||        |           |       Protection check       | Readable | Test | Writable | Test | Status   | Writable | Result | Status   | Writable | Result ||\n");
    dprintf("---------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    for (falconId=0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        dprintf("|| %s ", lsFalc[falconId].name);

        if (lsFalc[falconId].available == LW_FALSE)
        {
            dprintf("| Not supported                                                                                                                              ||\n");
            continue;
        }

        reg  = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK);
        reg1 = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_SCTL);

        if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, reg1) == LW_PFALCON_FALCON_SCTL_LSMODE_TRUE)
        {
            dprintf("| Yes       ");
            if (DRF_VAL(_PFALCON_FALCON, _DMEM_PRIV_LEVEL_MASK_, WRITE_PROTECTION_LEVEL0, reg)
                == LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK_WRITE_PROTECTION_LEVEL0_ENABLE)
            {
                dprintf("| No            ");
                writeProtection = LW_FALSE;
            }
            else
            {
                dprintf("| Yes           ");
                writeProtection = LW_TRUE;
            }
            if (DRF_VAL(_PFALCON_FALCON, _DMEM_PRIV_LEVEL_MASK_, READ_PROTECTION_LEVEL0, reg)
                == LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
            {
                dprintf("| No           ");
                readProtection = LW_FALSE;
            }
            else
            {
                dprintf("| Yes          ");
                readProtection = LW_TRUE;
            }

            cmd = FLD_SET_DRF(_PFALCON,_FALCON_DMEMC,_AINCR,_TRUE,0);
            GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index)),cmd);
            oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));
            reg    = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index));
            offset = DRF_VAL(_PFALCON_FALCON, _DMEMC, _OFFS, reg);

            if (readProtection)
            {
                if (offset == 0)
                {
                    dprintf("| No       | PASS ");
                }
                else
                {
                    dprintf("| Yes      | FAIL ");
                }
            }
            else
            {
                if (offset == 0)
                {
                    dprintf("| No       | FAIL ");
                }
                else
                {
                    dprintf("| Yes      | PASS ");
                }
            }

            if (readProtection)
            {
                cmd = FLD_SET_DRF(_PFALCON,_FALCON_DMEMC,_AINCW,_TRUE,0);
                GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index)),cmd);
                GPU_REG_WR32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index),0x12345678);
                reg    = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index));
                offset = DRF_VAL(_PFALCON_FALCON, _DMEMC, _OFFS, reg);

                if (writeProtection)
                {
                    if (offset == 0)
                    {
                        dprintf("| No       | PASS ");
                    }
                    else
                    {
                        dprintf("| Yes      | FAIL ");
                    }
                }
                else
                {
                    if (offset == 0)
                    {
                        dprintf("| No       | FAIL ");
                    }
                    else
                    {
                        dprintf("| Yes      | PASS ");
                    }
                }
            }
            else
            {
                oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                // Generating a new value by toggling bits;
                value = oldVal ^ 0xffffffff;

                GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index)),value);
                newVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                if (writeProtection)
                {
                    if (newVal == value)
                    {
                        dprintf("| Yes      | FAIL ");
                    }
                    else
                    {
                        dprintf("| No       | PASS ");
                    }
                }
                else
                {
                    if (newVal == value)
                    {
                        dprintf("| Yes      | PASS ");
                    }
                    else
                    {
                        dprintf("| No       | FAIL ");
                    }
                }
            }
        }
        else
        {
            dprintf("| No        | N/A           | N/A          | N/A      | N/A  | N/A      | N/A  ");
        }
        if (falconId == LSF_FALCON_ID_PMU || falconId == LSF_FALCON_ID_DPU)
        {
            if (DRF_VAL( _PFALCON_FALCON, _DMEM_PRIV_LEVEL_MASK, _READ_PROTECTION_LEVEL0,
                         GPU_REG_RD32(lsFalc[falconId].regBase + LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK))
                        == LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
            {
                for (range=0; range < 2; range++)
                {
                    switch (falconId)
                    {
                        case LSF_FALCON_ID_PMU:
                            if (range == 0)
                            {
                                reg2       = GPU_REG_RD32(LW_PPWR_FALCON_DMEM_PRIV_RANGE0);
                                rangeStart = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE0, _START_BLOCK, reg2);
                                rangeEnd   = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE0, _END_BLOCK, reg2);
                            }
                            else
                            {
                                reg2       = GPU_REG_RD32(LW_PPWR_FALCON_DMEM_PRIV_RANGE1);
                                rangeStart = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE1, _START_BLOCK, reg2);
                                rangeEnd   = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE1, _END_BLOCK, reg2);
                            }
                            break;
                        case LSF_FALCON_ID_DPU:
                            if (range == 0)
                            {
                                reg2       = GPU_REG_RD32(LW_PDISP_FALCON_DMEM_PRIV_RANGE0);
                                rangeStart = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE0, _START_BLOCK, reg2);
                                rangeEnd   = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE0, _END_BLOCK, reg2);
                            }
                            else
                            {
                                reg2       = GPU_REG_RD32(LW_PDISP_FALCON_DMEM_PRIV_RANGE1);
                                rangeStart = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE1, _START_BLOCK, reg2);
                                rangeEnd   = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE1, _END_BLOCK, reg2);
                            }
                            break;
                        default:
                            dprintf ("Invalid FalconId\n");
                            return LW_ERR_GENERIC;
                    }

                    if (rangeStart > rangeEnd)
                    {
                        dprintf("| Disabled | N/A      | N/A    ");
                        continue;
                    }
                    dprintf("| Enabled  ");

                    GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index)), (rangeStart << 8));

                    oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                    // Generating a new value by toggling bits;
                    value = oldVal ^ 0xffffffff;

                    GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index)),value);
                    newVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                    if (newVal == value)
                    {
                        dprintf("| Yes      | PASS   ");
                    }
                    else
                    {
                        dprintf("| No       | FAIL   ");
                    }
                }
                dprintf("||\n");
            }
            else
            {
                for (range=0; range < 2; range++)
                {
                    if (range == 0)
                    {
                        dprintf("| Enabled  ");
                        reg2 = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_HWCFG);
                        dmemSize = DRF_VAL( _PFALCON_FALCON, _HWCFG, _DMEM_SIZE, reg2);

                        // Checking at the end of range because the offset may change
                        rangeStart = (dmemSize << 8) - 0x10;

                        GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMC(index)), rangeStart);

                        oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                        // Generating a new value by toggling bits;
                        value = oldVal ^ 0xffffffff;

                        GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index)),value);
                        newVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_DMEMD(index));

                        if (newVal == value)
                        {
                            dprintf("| Yes      | PASS   ");
                        }
                        else
                        {
                            dprintf("| No       | FAIL   ");
                        }
                    }
                    else
                    {
                        dprintf("| Disabled | N/A      | N/A    ");
                        continue;
                    }
                }
                dprintf("||\n");
            }
        }
        else
        {
            dprintf("| N/A      | N/A      | N/A    | N/A      | N/A      | N/A    ||\n");
        }
    }
    dprintf("---------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    return status;
}

//-----------------------------------------------------
// acrImemProtection_GM20X - Verify IMEM protection
//-----------------------------------------------------
LW_STATUS acrImemProtection_GM20X( LwU32 indexGpu )
{
    LW_STATUS   status = LW_OK;
    LwU32  value;
    LwU32  oldVal;
    LwU32  newVal;
    LwU32  falconId;
    LwU32  reg;
    LwU32  reg1;
    LwU32  offset;
    LwU32  cmd;
    LwU32  index = 0;
    LwBool readProtection;
    LwBool writeProtection;

    dprintf("--------------------------------------------------------------------------------------------------------------------------------\n");
    dprintf("||  Falcon  || Is LSMode || Is LEVEL0 Write protected ? || Is LEVEL0 Read protected ? || Readable || Test || Writable || Test ||\n");
    dprintf("--------------------------------------------------------------------------------------------------------------------------------\n");

    for (falconId=0; falconId < LSF_FALCON_ID_END; falconId++)
    {
        dprintf("||  %s  ",lsFalc[falconId].name);

        if (lsFalc[falconId].available == LW_FALSE)
        {
            dprintf("|| Not supported                                                                                                  ||\n");
            continue;
        }

        reg  = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK);
        reg1 = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_SCTL);

        if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, reg1) == LW_PFALCON_FALCON_SCTL_LSMODE_TRUE)
        {
            dprintf("|| Yes       ");
            if (DRF_VAL(_PFALCON_FALCON, _IMEM_PRIV_LEVEL_MASK_, WRITE_PROTECTION_LEVEL0, reg)
                == LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK_WRITE_PROTECTION_LEVEL0_ENABLE)
            {
                dprintf("|| No                          ");
                writeProtection = LW_FALSE;
            }
            else
            {
                dprintf("|| Yes                         ");
                writeProtection = LW_TRUE;
            }
            if (DRF_VAL(_PFALCON_FALCON, _IMEM_PRIV_LEVEL_MASK_, READ_PROTECTION_LEVEL0, reg)
                == LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
            {
                dprintf("|| No                         ");
                readProtection = LW_FALSE;
            }
            else
            {
                dprintf("|| Yes                        ");
                readProtection = LW_TRUE;
            }

            cmd = FLD_SET_DRF(_PFALCON,_FALCON_IMEMC,_AINCR,_TRUE,0);
            GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMC(index)),cmd);
            oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMD(index));
            reg    = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMC(index));
            offset = DRF_VAL(_PFALCON_FALCON, _IMEMC, _OFFS, reg);

            if (readProtection)
            {
                if (offset == 0)
                {
                    dprintf("|| No       || PASS ");
                }
                else
                {
                    dprintf("|| Yes      || FAIL ");
                }
            }
            else
            {
                if (offset == 0)
                {
                    dprintf("|| No       || FAIL ");
                }
                else
                {
                    dprintf("|| Yes      || PASS ");
                }
            }

            if (readProtection)
            {
                cmd = FLD_SET_DRF(_PFALCON,_FALCON_IMEMC,_AINCW,_TRUE,0);
                GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMC(index)),cmd);
                GPU_REG_WR32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMD(index),0x12345678);
                reg    = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMC(index));
                offset = DRF_VAL(_PFALCON_FALCON, _IMEMC, _OFFS, reg);

                if (writeProtection)
                {
                    if (offset == 0)
                    {
                        dprintf("|| No       || PASS ||\n");
                    }
                    else
                    {
                        dprintf("|| Yes      || FAIL ||\n");
                    }
                }
                else
                {
                    if (offset == 0)
                    {
                        dprintf("|| No       || FAIL ||\n");
                    }
                    else
                    {
                        dprintf("|| Yes      || PASS ||\n");
                    }
                }
            }
            else
            {
                oldVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMD(index));

                // Generating a new value by toggling bits;
                value = oldVal ^ 0xffffffff;

                GPU_REG_WR32((lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMD(index)),value);
                newVal = GPU_REG_RD32(lsFalc[falconId].regBase+LW_PFALCON_FALCON_IMEMD(index));

                if (writeProtection)
                {
                    if (newVal == value)
                    {
                        dprintf("|| Yes      || FAIL ||\n");
                    }
                    else
                    {
                        dprintf("|| No       || PASS ||\n");
                    }
                }
                else
                {
                    if (newVal == value)
                    {
                        dprintf("|| Yes      || PASS ||\n");
                    }
                    else
                    {
                        dprintf("|| No       || FAIL ||\n");
                    }
                }
            }
        }
        else
        {
            dprintf("|| No        || N/A                         || N/A                        || N/A      || N/A  || N/A      || N/A  ||\n");
        }
    }
    dprintf("--------------------------------------------------------------------------------------------------------------------------------\n");
    return status;
}

void acrGetFalconProp_GM20X(LSFALCPROP *pFalc, LwU32  falconId, LwU32 indexGpu)
{
    switch(falconId)
    {
        case LSF_FALCON_ID_PMU:
                pFalc[falconId].name            = "PMU   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_PWR_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PWR), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_DPU:
                pFalc[falconId].name            = "DPU   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_DISP_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PDISP), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_FECS:
                pFalc[falconId].name            = "FECS  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_FECS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_GPCCS:
                pFalc[falconId].name            = "GPCCS ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_GPC0_GPCCS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWDEC:
                pFalc[falconId].name            = "LWDEC ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWDEC_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWDEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC0:
                pFalc[falconId].name            = "LWENC0";
                pFalc[falconId].regBase         = LW_FALCON_LWENC0_BASE;
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC0), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC1:
                pFalc[falconId].name            = "LWENC1";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWENC1_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC1), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_SEC2:
                pFalc[falconId].name            = "SEC   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PSEC_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_SEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC2:
                pFalc[falconId].name            = "LWENC2";
                pAcr[indexGpu].acrGetLwenc2FalconProp(pFalc);
                break;
    }
}

BOOL acrIsFmodel_GM20X (void)
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
