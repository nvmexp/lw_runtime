/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2016 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//
// includes
//

#include "maxwell/gm107/hwproject.h"
#include "chip.h"

// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/
#include "maxwell/gm107/hwproject.h"
#include "maxwell/gm107/dev_top.h"
#include "maxwell/gm107/dev_fuse.h"
#include "maxwell/gm107/dev_graphics_nobundle.h"
#include "maxwell/gm107/dev_fifo.h"

#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

#define SM_DFD_CONTROL_WARP_PC_AUTO_INCREMENT_READ 0x07000000
#define SM_DFD_CONTROL_WARP_PC_ITERATE_SUBPARTITION_THEN_ROW 0x000B0000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_UPPER 0x00004000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_LOWER 0x00003000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP 0x00000F00
#define SM_DFD_CONTROL_WARP_PC_STARTING_SCH_ISSUE_PC 0x00000030
#define SM_DFD_CONTROL_WARP_PC_STARTING_SUBPARTITION_WARP 0x00000000


LwU32 grGetNumberPesPerGpc_GM107(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetMaxGpc_GM107()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GM107(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

LwU32 grGetMaxTpcPerGpc_GM107()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

/*!
 * Colwerts the "disable_mask" partition FS config
 * into "enable_mask" partition FS config.
 */
static LwU32 extractAndIlwert ( LwU32 nActive, LwU32 nMax )
{
    LwU32 mask = BITMASK (nMax);
    return (( ~nActive ) & mask);
}

// Gets active GPC config.
LwBool grGetActiveGpcConfig_GM107(LwU32 *activeGpcConfig, LwU32 *maxNumberOfGpcs)
{
    *activeGpcConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_GPC);
    if ((*activeGpcConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_GPC register read gave 0x%x value.\n", *activeGpcConfig);
        return LW_FALSE;
    }
    *maxNumberOfGpcs = GPU_REG_RD32(LW_PTOP_SCAL_NUM_GPCS);
    if ((*maxNumberOfGpcs & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_GPCS register read gave 0x%x value.\n", *maxNumberOfGpcs);
        return LW_FALSE;
    }
    *activeGpcConfig = extractAndIlwert (*activeGpcConfig, *maxNumberOfGpcs);
    return LW_TRUE;
}

// Gets active TPC config for GpcId.
LwBool grGetActiveTpcConfig_GM107(LwU32 GpcId, LwU32 *activeTpcConfig, LwU32 *maxNumberOfTpcs)
{
    *activeTpcConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_TPC_GPC(GpcId));
    if ((*activeTpcConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_TPC(%u) register read gave 0x%x value.\n", GpcId, *activeTpcConfig);
        return LW_FALSE;
    }
    *maxNumberOfTpcs = GPU_REG_RD32(LW_PTOP_SCAL_NUM_TPC_PER_GPC);
    if ((*maxNumberOfTpcs & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_TPC_PER_GPC register read gave 0x%x value.\n", *maxNumberOfTpcs);
        return LW_FALSE;
    }
    *activeTpcConfig = extractAndIlwert (*activeTpcConfig,*maxNumberOfTpcs);
    return LW_TRUE;
}

// Gets active ZLWLL config for GpcId.
LwBool grGetActiveZlwllConfig_GM107(LwU32 GpcId, LwU32 *activeZlwllConfig, LwU32 *maxNumberOfZlwlls)
{
    *activeZlwllConfig = GPU_REG_RD32(LW_FUSE_OPT_ZLWLL_DISABLE);
    if ((*activeZlwllConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_ZLWLL(%u) register read gave 0x%x value.\n", GpcId, *activeZlwllConfig);
        return LW_FALSE;
    }

    *maxNumberOfZlwlls = GPU_REG_RD32(LW_PTOP_SCAL_NUM_ZLWLL_BANKS);
    if ((*maxNumberOfZlwlls & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_ZLWLL_BANKS register read gave 0x%x value.\n", *maxNumberOfZlwlls);
        return LW_FALSE;
    }

    *activeZlwllConfig = *activeZlwllConfig >> (GpcId * (*maxNumberOfZlwlls));
    *activeZlwllConfig = extractAndIlwert (*activeZlwllConfig,*maxNumberOfZlwlls);
    return LW_TRUE;
}

// Gets active FBP config.
LwBool grGetActiveFbpConfig_GM107(LwU32 *activeFbpConfig, LwU32 *maxNumberOfFbps)
{
    *activeFbpConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_FBP);
    if ((*activeFbpConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_FBP register read gave 0x%x value.\n", *activeFbpConfig);
        return LW_FALSE;
    }
    *maxNumberOfFbps = GPU_REG_RD32(LW_PTOP_SCAL_NUM_FBPS);
    if ((*maxNumberOfFbps & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_FBPS register read gave 0x%x value.\n", *maxNumberOfFbps);
        return LW_FALSE;
    }
    *activeFbpConfig = extractAndIlwert (*activeFbpConfig,*maxNumberOfFbps);
    return LW_TRUE;
}

// Gets active FBP config.
LwBool grGetActiveFbpaConfig_GM107(LwU32 *activeFbpaConfig, LwU32 *maxNumberOfFbpas)
{
    *activeFbpaConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_FBPA);
    if ((*activeFbpaConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_FBPA register read gave 0x%x value.\n", *activeFbpaConfig);
        return LW_FALSE;
    }
    *maxNumberOfFbpas = GPU_REG_RD32(LW_PTOP_SCAL_NUM_FBPAS);
    if ((*maxNumberOfFbpas & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_FBPAS register read gave 0x%x value.\n", *maxNumberOfFbpas);
        return LW_FALSE;
    }
    *activeFbpaConfig = extractAndIlwert (*activeFbpaConfig,*maxNumberOfFbpas);
    return LW_TRUE;
}

// Gets active LWENC config.
LwBool grGetActiveLwencConfig_GM107(LwU32 *activeLwencConfig, LwU32 *maxNumberOfLwencs)
{
    *activeLwencConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC);
    if ((*activeLwencConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_LWENC register read gave 0x%x value.\n", *activeLwencConfig);
        return LW_FALSE;
    }
    *maxNumberOfLwencs = GPU_REG_RD32(LW_PTOP_SCAL_NUM_LWENCS);
    if ((*maxNumberOfLwencs & 0xFFFF0000) == 0xBADF0000) {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_LWENCS register read gave 0x%x value.\n", *maxNumberOfLwencs);
        return LW_FALSE;
    }
    *activeLwencConfig = extractAndIlwert (*activeLwencConfig,*maxNumberOfLwencs);
    return LW_TRUE;
}

// Gets active CE config.
LwBool grGetActiveCeConfig_GM107(LwU32 *activeCeConfig, LwU32 *maxNumberOfCes)
{
    *activeCeConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_CE);
    if ((*activeCeConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_CE register read gave 0x%x value.\n", *activeCeConfig);
        return LW_FALSE;
    }
    *maxNumberOfCes = GPU_REG_RD32(LW_PTOP_SCAL_NUM_CES);
    if ((*maxNumberOfCes & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_CES register read gave 0x%x value.\n", *maxNumberOfCes);
        return LW_FALSE;
    }
    *activeCeConfig = extractAndIlwert (*activeCeConfig,*maxNumberOfCes);
    return LW_TRUE;
}

// Gets active DISP_HEAD config.
LwBool grGetActiveDispHeadConfig_GM107(LwU32 *activeDispHeadConfig, LwU32 *maxNumberOfDispHeads)
{
    *activeDispHeadConfig = GPU_REG_RD32(LW_FUSE_STATUS_OPT_DISP_HEAD);
    if ((*activeDispHeadConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_DISP_HEAD register read gave 0x%x value.\n", *activeDispHeadConfig);
        return LW_FALSE;
    }
    *maxNumberOfDispHeads = GPU_REG_RD32(LW_PTOP_SCAL_NUM_DISP_HEADS);
    if ((*maxNumberOfDispHeads & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_DISP_HEADS register read gave 0x%x value.\n", *maxNumberOfDispHeads);
        return LW_FALSE;
    }
    *activeDispHeadConfig = extractAndIlwert (*activeDispHeadConfig,*maxNumberOfDispHeads);
    return LW_TRUE;
}

///Insert warping function

// 
// On Maxwell the Warp PC's can be printed via the DFD machine, so
// HALifying TPC information printout since those registers aren't
// define for Kepler.  Before checking in, check with sw-lwwatch to
// see if this is the best approach.
// 
// This method is called for all GM10x and GM20x chips.

static void _grDumpTpcWarpPcs( LwU32 gpcId, LwU32 tpcId, LwU32 repeatPerWarp, LwBool annotate )
{
    // Used by gr.h print macros
    char buffer[GR_REG_NAME_BUFFER_LEN];

    LwU32 warpPc, warpId, subPartitionId, maskId, smWarpId;
    LwU32 warpValidMaskPtr[4];
    LwU32 nSubPartitions = 4, warpValid = 0;
    LwU32 *warpPcBuffer = NULL;
    LwU32 repeat = 0;
    LwU32 dfdCommand = 0;

    // Get the number of possible warps
    LwU32 warpCount = DRF_VAL (_PGRAPH, _PRI_GPC0_TPC0_SM_ARCH, _WARP_COUNT, GPU_REG_RD32 (TPC_REG_ADDR (SM_ARCH, gpcId, tpcId)));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_ARCH_WARP_COUNT = %u\n", gpcId, tpcId, warpCount);
    
    if (repeatPerWarp)
    {
        warpPcBuffer = (LwU32 *) calloc (repeatPerWarp * warpCount, sizeof (LwU32));
        if ( warpPcBuffer == NULL )
        {
            dprintf ("Error! Memory could not be allocated.");
            return;
        }
    }

    warpValidMaskPtr [0] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_0, gpcId, tpcId));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_0 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_0, gpcId, tpcId), warpValidMaskPtr [0]);
    warpValid |= warpValidMaskPtr [0];
    warpValidMaskPtr [1] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_1, gpcId, tpcId));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_1 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_1, gpcId, tpcId), warpValidMaskPtr [1]);
    warpValid |= warpValidMaskPtr [1];



    if (warpValid)
    {
        if (repeatPerWarp)
        {
            char *inactiveWarp = " (i)";
            
            dprintf("Warp PCs for GPC %d, TPC %d, (i) == inactive warp\n", gpcId, tpcId );
            if (!annotate)
            {
                dprintf("%14s %10s %10s\n", "Subpartition", "Warp ID", "PC" );
            }

            for ( repeat = 0; repeat < repeatPerWarp; repeat += 1)
            {
                // Turn on the DFD machine and set up the warp PC read
                // pattern.  Only print PC for active warps, but do the read
                // for every warp so the DFD auto-increment happens.
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), dfdCommand);
                for ( warpId = 0 ; warpId < (warpCount / nSubPartitions) ; warpId ++ )
                {
                    for ( subPartitionId = 0 ; subPartitionId < nSubPartitions ; subPartitionId ++ )
                    {
                        smWarpId = warpId * nSubPartitions + subPartitionId;

                        warpPc = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpPcBuffer[smWarpId * repeatPerWarp + repeat] = warpPc;
                    }
                }
                // Shut down DFD machine
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), 0x0);
            }

            for ( warpId = 0 ; warpId < (warpCount / nSubPartitions) ; warpId ++ )
            {
                for ( subPartitionId = 0 ; subPartitionId < nSubPartitions ; subPartitionId ++ )
                {
                    LwBool warpIsValid;
                    smWarpId = warpId * nSubPartitions + subPartitionId;
                    maskId = smWarpId / 32;
                    warpIsValid = (warpValidMaskPtr[maskId] & 0x1) != 0;

                    if (annotate)
                    {
                        dprintf("GPC %2u, TPC %2u, warp %4u ", gpcId, tpcId, smWarpId);
                    }
                    else
                    {
                        dprintf("%14d %10d", subPartitionId, warpId);
                    }


                    for ( repeat = 0; repeat < repeatPerWarp; repeat += 1 )
                    {
                        if (repeat && repeat % 4 == 0)
                        {
                            dprintf("\n%26s", " ");
                        }
                        warpPc = warpPcBuffer[smWarpId * repeatPerWarp + repeat];
                        dprintf(" 0x%.8x", warpPc);
                    }
                
                    dprintf("%s\n", warpIsValid ? "" : inactiveWarp );
                    warpValidMaskPtr [maskId] >>= 1;
                }
            }
        }

        PRINT_TPC_REG_PD ( SM_DBGR_BPT_PAUSE_MASK_0, gpcId, tpcId );
        PRINT_TPC_REG_PD ( SM_DBGR_BPT_PAUSE_MASK_1, gpcId, tpcId );
        PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_0, gpcId, tpcId );
        PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_1, gpcId, tpcId );
    }

    if (warpPcBuffer)
    {
        free (warpPcBuffer);
    }
}




// 
// On Maxwell the Warp PC's can be printed via the DFD machine, so
// HALifying TPC information printout since those registers aren't
// define for Kepler.  Before checking in, check with sw-lwwatch to
// see if this is the best approach.
// 
// This method is called for all GM10x and GM20x chips.

void grDumpTpcInfo_GM107(LwU32 gpcId, LwU32 tpcId)
{   
    char buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 data32, addr32;
    buffer[0] = '\0';
    
    dprintf("====================\n");
    dprintf("%s GPC/TPC %d/%d\n", GpuArchitecture(), gpcId, tpcId);
    dprintf("====================\n");

    PRINT_TPC_REG_PD(TPCCS_TPC_ACTIVITY0, gpcId, tpcId);

    PRINT_TPC_REG_PD(PE_STATUS, gpcId, tpcId);

    DUMP_TPC_REG(PE_L2_EVICT_POLICY, gpcId, tpcId);
    DUMP_TPC_REG(PE_HWW_ESR, gpcId, tpcId);

    // These registers hold 3 bit values in ACTIVITY format but those
    // encodings aren't enumerated in the manuals, so priv_dump just
    // prints the numeric values.
    PRINT_TPC_REG_PD(MPC_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_VTG_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_PIX_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_WLU_STATUS, gpcId, tpcId);
 
    _grDumpTpcWarpPcs( gpcId, tpcId, 1, LW_FALSE );

    // Implementation of bug #1374564
    // TEX PRIs route to PIPE 0 by default
    
    dprintf ("\n==========\nTEX PIPE 0\n==========\n");
    
    addr32 = TPC_REG_ADDR( TEX_M_ROUTING, gpcId, tpcId );
    data32 = GPU_REG_RD32( addr32 );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
        LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
    GPU_REG_WR32( addr32, data32 );

    DUMP_TPC_REG(TEX_M_ROUTING, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_REQ, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_ADDR, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_ADDR1, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_MMU, gpcId, tpcId);
    PRINT_TPC_REG_PD(TEX_M_TEX_SUBUNITS_STATUS, gpcId, tpcId);

    // Route to PIPE 1
    
    dprintf ("\n==========\nTEX PIPE 1\n==========\n");
    
    addr32 = TPC_REG_ADDR( TEX_M_ROUTING, gpcId, tpcId );
    data32 = GPU_REG_RD32( addr32 );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
        LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1 );
    GPU_REG_WR32( addr32, data32 );
    
    DUMP_TPC_REG(TEX_M_ROUTING, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_REQ, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_ADDR, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_ADDR1, gpcId, tpcId);
    DUMP_TPC_REG(TEX_M_HWW_ESR_MMU, gpcId, tpcId);
    PRINT_TPC_REG_PD(TEX_M_TEX_SUBUNITS_STATUS, gpcId, tpcId);

    dprintf ("\n");

    DUMP_TPC_REG(L1C_HWW_ESR, gpcId, tpcId);

    PRINT_TPC_REG_PD(L1C_CACHE_MGMT_CSR, gpcId, tpcId);

    DUMP_TPC_REG(L1C_ECC_ERROR, gpcId, tpcId);
    DUMP_TPC_REG(L1C_ECC_ADDRESS, gpcId, tpcId);

    PRINT_TPC_REG_PD ( SM_BLK_ACTIVITY_PRIV_LEVEL_MASK, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG2, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_TEXIO_CONTROL, gpcId, tpcId );
    DUMP_TPC_REG(SM_HWW_ESR_ADDR_0, gpcId, tpcId);
    DUMP_TPC_REG(SM_HWW_ESR_ADDR_1, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_WARP_ESR_REPORT_MASK, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_WARP_ESR, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_GLOBAL_ESR_REPORT_MASK, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_GLOBAL_ESR, gpcId, tpcId);


    DUMP_TPC_REG(SM_CONFIG, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_ARCH, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_CACHE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_DISP_CTRL, gpcId, tpcId);
    DUMP_TPC_REG(SM_MACHINE_ID0, gpcId, tpcId);

    DUMP_TPC_REG(SM_LRF_ECC_SINGLE_ERR_COUNT, gpcId, tpcId);
    DUMP_TPC_REG(SM_LRF_ECC_DOUBLE_ERR_COUNT, gpcId, tpcId);
    DUMP_TPC_REG(SM_LRF_ECC_ADDRESS, gpcId, tpcId);

    PRINT_TPC_REG_PD ( SM_DEBUG_SFE_CONTROL, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_QUAD_BA_CONTROL, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_POWER_THROTTLE_CTL, gpcId, tpcId );

    // CYA bits in the SM LW_PGRAPH_PRI_GPC#0_TPC#_SM_HALFCTL_CTRL
    PRINT_TPC_REG_PD( SM_HALFCTL_CTRL, gpcId, tpcId );

    // Subunit and subunit_half status
    PRINT_TPC_REG_PD( SM_INFO_SUBUNIT_STATUS, gpcId, tpcId);

    // SUBUNIT_HALF_STATUS is muxed, need to write the partition
    // select before reading.

    // Base address for control register in this gpc/tpc
    addr32 = TPC_REG_ADDR( SM_HALFCTL_CTRL, gpcId, tpcId );

    // Set up to read partition 0
    data32 = GPU_REG_RD32( addr32 );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_SM_HALFCTL_CTRL, _SCTL_READ_QUAD_CTL,
        LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL_0 );
    GPU_REG_WR32( addr32, data32 );

    // Dump partition 0
    dprintf("Partition 0 SUBUNIT_HALF_STATUS\n");
    PRINT_TPC_REG_PD(SM_INFO_SUBUNIT_HALF_STATUS, gpcId, tpcId);

    // Set up to read partition 1
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_SM_HALFCTL_CTRL, _SCTL_READ_QUAD_CTL,
        LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL_1 );
    GPU_REG_WR32( addr32, data32 );

    // Dump partition 1
    dprintf("Partition 1 SUBUNIT_HALF_STATUS\n");
    PRINT_TPC_REG_PD(SM_INFO_SUBUNIT_HALF_STATUS, gpcId, tpcId);


    // Reset default partition select
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_SM_HALFCTL_CTRL, _SCTL_READ_QUAD_CTL,
        LW_PGRAPH_PRI_GPC0_TPC0_SM_HALFCTL_CTRL_SCTL_READ_QUAD_CTL_INIT );
    GPU_REG_WR32( addr32, data32 );
}


/*----------------------------------------------------------------------------
 * static void
 * grDumpConsolidatedReport_GM107()
 *      Print a tightly formatted report of chip state, allowing quick
 *      review by SW teams when determining if a bug needs to be
 *      looked at by HW, and by HW teams when determining where to
 *      look next.
 *
 * Return Value --
 *      void.
 *
 *----------------------------------------------------------------------------
 */

void grDumpConsolidatedReport_GM107( LwU32 grIdx )
{
    LwU32 grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, GK104);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GM107 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GK104 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GM107 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GK104 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    pgraphStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
    act4 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY4);

    // For each unit, if it'd not IDLE in LW_PGRAPH_STATUS, print the
    // associated activity value from LW_PGRAPH_ACTIVITY*

    DUMP_REG(ACTIVITY0);        /* GK104 and GM107 are the same */
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PD, act0),
                        "PD");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PDB, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PDB, act0),
                        "PDB");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SCC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SCC, act0),
                        "SCC");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _RASTWOD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _RASTWOD, act0),
                        "RASTWOD");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SSYNC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SSYNC, act0),
                        "SSYNC");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _CWD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _CWD, act0),
                        "CWD");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SKED, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SKED, act0),
                        "SKED");
    }


    DUMP_REG(ACTIVITY1);        /* GK104 and GM107 are the same */

    pGr[indexGpu].grDumpConsolidatedReportPMM(pgraphStatus, act1);

    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _MEMFMT, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _MEMFMT, act1),
                        "MEMFMT");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _SEMAPHORE, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _SEMAPHORE, act1),
                        "SEMAPHORE");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_FUNNEL, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FUNNEL, act1),
                        "FE_FUNNEL");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_CONST, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FECONST, act1),
                        "FECONST");
    }


    // Per FBP
    DUMP_REG(ACTIVITY2);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act2;
        for (i=0;i<fbpCount;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY2, _BE0, fbpStatus), regName);
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY2_BE0);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GF100 );
        }
    }

    // Per GPC
    DUMP_REG(ACTIVITY4);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _GPC, _BUSY))
    {
        LwU32 i;
        LwU32 gpcStatus = act4;
        for (i=0;i<gpcCount;i++)
        {
            sprintf(regName, "GPC%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL(_PGRAPH, _ACTIVITY4, _GPC0, gpcStatus), regName);
            gpcStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY4_GPC0);
        }
        pGr[indexGpu].grDumpConsolidatedReportGpc( gpcCount, grIdx );
    }
}

void grDumpConsolidatedReportPMM_GM107(LwU32 grStatus, LwU32 act1)
{
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PMA, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMA, act1),
                        "PMA");
    }

    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PMMSYS, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMMSYS, act1),
                        "PMMSYS");
    }
}

void grDumpConsolidatedReportGpc_GM107( LwU32 numActiveGpc, LwU32 grIdx )
{
    LwU32 tpcId, numActiveTpc, gpcId, grStatus, val;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        dprintf("====================\n");
        dprintf("GPC %d\n", gpcId);
        dprintf("====================\n");

        numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY0, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY1, GF100 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY2, GM107 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY3, GF100 );

        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_1, GM107 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_GPC_0, GF100 );

        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {
            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_REG_Z( gpcId, tpcId, TPCCS_TPC_ACTIVITY0, GM107 );
            PRINT_TPC_REG_Z( gpcId, tpcId, PE_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_VTG_STATUS, GM107 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_PIX_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, SM_INFO_SUBUNIT_STATUS, GM107 );
        }
    }
}

void grDumpGrfifoStatusPtrInfo_GM107(LwU32 data32)
{
    dprintf(" +\t%-30s\t: 0x%x\n",
            "_COUNT",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _COUNT, data32));

    dprintf(" +\t%-30s\t: 0x%x\n",
            "_READ_PTR",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _READ_PTR, data32));

    dprintf(" +\t%-30s\t: 0x%x\n",
            "_WRITE_PTR",
            DRF_VAL(_PGRAPH, _GRFIFO_STATUS, _WRITE_PTR, data32));
}

//no FE Semaphore Acquire registers starting from MAXWELL
LW_STATUS grDumpPriFeSemaphoreState_GM107(void)
{
    LW_STATUS status = LW_OK;

    dprintf("====================\n");
    dprintf(" FE SEMAPHORE STATE detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_A);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_B);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_C);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_D);
    PRINT_REG_PD(_PGRAPH_PRI_FE, _SEMAPHORE_STATE_REPORT);

    return status;
}

// LW_PFIFO_ENGINE_STATUS() has a different stride starting on Kepler
void grPrintEngineGraphicsStatus_GM107(void)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];

    // A little tricky: print address of this reg as a string in order
    // to use priv_dump, which will print all of the fields
    // automatically.  It won't print the fact that _ENGINE_GRAPHICS
    // is the register being shown (most likely the value will be 0,
    // and not defined from Maxwell through Turing, so putting the value),
    // so print that beforehand to make it clear.  priv_dump will
    // recognize "LW_PFIFO_ENGINE_STATUS(0)" as an address, but not
    // "LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS)".  Using them
    // this way expands and checks the values at compile time.
    dprintf("LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS):\n");
    sprintf( buffer, "0x%08x", LW_PFIFO_ENGINE_STATUS(0) );
    priv_dump( buffer );
}

