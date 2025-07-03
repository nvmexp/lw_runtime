/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2019 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//
// includes
//

#include "maxwell/gm200/hwproject.h"
#include "chip.h"
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/<family>/<chip>
#include "maxwell/gm200/dev_graphics_nobundle.h"
#include "maxwell/gm200/dev_tpc.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

// Local defines.  These DFD defines should be manuals
// See http://lwbugs/865794/92 (pascal), http://lwbugs/865794/93
// (maxwell), http://lwbugs/1294327, http://lwbugs/865794/35 for
// SM_DFD_CONTROL programming example.
// - 0x07 is the opcode to program auto increment read
// - 0x0B programs the iterators (which "axis" to increment). 0xB
//   means more across the subpartitions and then increment the "row".
// - 0x4F is the ending "row", 0x30 is the starting "row"
// Explanation on "rows":
// - Row[7] : selects between SCH vs all other subunits.  SCH has a lot of states (like the PC) so it requires 6 bits of sub selects
// --- SCH = (Row[7] == 0)
// - Row[6:4] : selects the state groups within the SCH
// --- lower PC group (warp 0 -15 within the subpartition),
//     this is the same as Maxwell
// --- upper PC group (warp 16-31 within the subpartition),
//     this is the expanded portion for GP100 and GP10b but also
//     applicable for GP10x.  For GP10x, this will read out 0x1 to
//     indicate invalid/inactive warps
// - Row[3:0] : selects the item within the state group
// --- For the PC state group, this is the warp ID

#define SM_DFD_CONTROL_WARP_PC_AUTO_INCREMENT_READ 0x07000000
#define SM_DFD_CONTROL_WARP_PC_ITERATE_SUBPARTITION_THEN_ROW 0x000B0000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_UPPER 0x00004000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_LOWER 0x00003000
#define SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP 0x00000F00
#define SM_DFD_CONTROL_WARP_PC_STARTING_SCH_ISSUE_PC 0x00000030
#define SM_DFD_CONTROL_WARP_PC_STARTING_SUBPARTITION_WARP 0x00000000

#define SM_DFD_CONTROL_WARP_PC_READ_128WARPS                    \
      SM_DFD_CONTROL_WARP_PC_AUTO_INCREMENT_READ            |   \
      SM_DFD_CONTROL_WARP_PC_ITERATE_SUBPARTITION_THEN_ROW  |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_UPPER      |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP       |   \
      SM_DFD_CONTROL_WARP_PC_STARTING_SCH_ISSUE_PC          |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP 

#define SM_DFD_CONTROL_WARP_PC_READ_64WARPS                     \
      SM_DFD_CONTROL_WARP_PC_AUTO_INCREMENT_READ            |   \
      SM_DFD_CONTROL_WARP_PC_ITERATE_SUBPARTITION_THEN_ROW  |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SCH_ISSUE_PC_LOWER      |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP       |   \
      SM_DFD_CONTROL_WARP_PC_STARTING_SCH_ISSUE_PC          |   \
      SM_DFD_CONTROL_WARP_PC_ENDING_SUBPARTITION_WARP 


LwU32 grGetNumberPesPerGpc_GM200(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetMaxGpc_GM200()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GM200(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

LwU32 grGetMaxTpcPerGpc_GM200()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

// 
// On Maxwell the Warp PC's can be printed via the DFD machine, so
// HALifying TPC information printout since those registers aren't
// define for Kepler.  Before checking in, check with sw-lwwatch to
// see if this is the best approach.
// 
// This method is called for all GM10x and GM20x chips.
// SM_DBGR_.*_MASK_2 and SM_DBGR_.*_MASK3 registers present only for GM20x onwards.
// So, these are guarded using IsGM200orLater ().

static void _grDumpTpcWarpPcs( LwU32 gpcId, LwU32 tpcId, LwU32 repeatPerWarp, LwBool bAnnotate )
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

    if ((IsGM200orLater ()) && ( warpCount == 128 ))
    {
        dfdCommand = SM_DFD_CONTROL_WARP_PC_READ_128WARPS;
    }
    else
    {
        dfdCommand = SM_DFD_CONTROL_WARP_PC_READ_64WARPS;
    }
    
    warpValidMaskPtr [0] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_0, gpcId, tpcId));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_0 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_0, gpcId, tpcId), warpValidMaskPtr [0]);
    warpValid |= warpValidMaskPtr [0];
    warpValidMaskPtr [1] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_1, gpcId, tpcId));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_1 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_1, gpcId, tpcId), warpValidMaskPtr [1]);
    warpValid |= warpValidMaskPtr [1];

    // For chips with 128 warps, additionally read masks 2 and 3
    if ((IsGM200orLater ()) && ( warpCount == 128 ))
    {
        warpValidMaskPtr [2] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_2, gpcId, tpcId));
        dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_2 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR (SM_DBGR_WARP_VALID_MASK_2, gpcId, tpcId), warpValidMaskPtr [2]);
        warpValid |= warpValidMaskPtr [2];
        warpValidMaskPtr [3] = GPU_REG_RD32( TPC_REG_ADDR( SM_DBGR_WARP_VALID_MASK_3, gpcId, tpcId));
        dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_DBGR_WARP_VALID_MASK_3 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, TPC_REG_ADDR (SM_DBGR_WARP_VALID_MASK_3, gpcId, tpcId), warpValidMaskPtr [3]);
        warpValid |= warpValidMaskPtr [3];
    }

    if (warpValid)
    {
        if (repeatPerWarp)
        {
            char *inactiveWarp = " (i)";
            
            dprintf("Warp PCs for GPC %d, TPC %d, (i) == inactive warp\n", gpcId, tpcId );
            if (!bAnnotate)
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

                    if (bAnnotate)
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
        if ((IsGM200orLater ()) && ( warpCount == 128 ))
        {
            PRINT_TPC_REG_PD ( SM_DBGR_BPT_PAUSE_MASK_2, gpcId, tpcId );
            PRINT_TPC_REG_PD ( SM_DBGR_BPT_PAUSE_MASK_3, gpcId, tpcId );
        }

        PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_0, gpcId, tpcId );
        PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_1, gpcId, tpcId );
        if ((IsGM200orLater ()) && ( warpCount == 128 ))
        {
            PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_2, gpcId, tpcId );
            PRINT_TPC_REG_PD ( SM_DBGR_BPT_TRAP_MASK_3, gpcId, tpcId );
        }
    }

    if (warpPcBuffer)
    {
        free (warpPcBuffer);
    }
}

void grDumpWarpPc_GM200(LwU32 gpcId, LwU32 tpcId, LwU32 repeat, LwBool bAnnotate)
{
    _grDumpTpcWarpPcs(gpcId, tpcId, repeat, bAnnotate);
}

// This method is called for all GM10x and GM20x chips.
// Registers that are present in GM20x but not in GM10x are guarded by IsGM200orLater ().

void grDumpTpcInfo_GM200(LwU32 gpcId, LwU32 tpcId)
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
    PRINT_TPC_REG_PD(MPC_COMP_STATUS, gpcId, tpcId);

    pGr[indexGpu].grDumpWarpPc( gpcId, tpcId, 1, LW_FALSE );

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
    if (IsGM200orLater ()) {
        PRINT_TPC_REG_PD ( SM_PM_SAMP_CTRL, gpcId, tpcId );
    }
    DUMP_TPC_REG(SM_HWW_ESR_ADDR_0, gpcId, tpcId);
    DUMP_TPC_REG(SM_HWW_ESR_ADDR_1, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_WARP_ESR_REPORT_MASK, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_WARP_ESR, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_GLOBAL_ESR_REPORT_MASK, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_HWW_GLOBAL_ESR, gpcId, tpcId);

    DUMP_TPC_REG(SM_CONFIG, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_ARCH, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_CACHE_CONTROL, gpcId, tpcId);
    if (IsGM200orLater ()) {
        PRINT_TPC_REG_PD(SM_PRIVATE_CONTROL, gpcId, tpcId);
    }
    PRINT_TPC_REG_PD(SM_DISP_CTRL, gpcId, tpcId);
    DUMP_TPC_REG(SM_MACHINE_ID0, gpcId, tpcId);
    DUMP_TPC_REG(SM_LRF_ECC_SINGLE_ERR_COUNT, gpcId, tpcId);
    DUMP_TPC_REG(SM_LRF_ECC_DOUBLE_ERR_COUNT, gpcId, tpcId);
    DUMP_TPC_REG(SM_LRF_ECC_ADDRESS, gpcId, tpcId);

    PRINT_TPC_REG_PD ( SM_DEBUG_SFE_CONTROL, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_QUAD_BA_CONTROL, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_POWER_THROTTLE_CTL, gpcId, tpcId );
    if (IsGM200orLater ()) {
        PRINT_TPC_REG_PD ( SM_POWER_VST_CTL, gpcId, tpcId );
        PRINT_TPC_REG_PD ( SM_POWER_VST_DATA, gpcId, tpcId );
    }

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


void grDumpConsolidatedReport_GM200( LwU32 grIdx )
{
    LwU32   grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act3, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, GM200);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GM107 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GK104 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GM200 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GK104 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    pgraphStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
    act3 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY3);
    act4 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY4);

    // For each unit, if it'd not IDLE in LW_PGRAPH_STATUS, print the
    // associated activity value from LW_PGRAPH_ACTIVITY*

    DUMP_REG(ACTIVITY0);       
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


    DUMP_REG(ACTIVITY1);  


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
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _TPC_MGR, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _TPCMGR, act1),
                        "TPCMGR");
    }


    // Per FBP
    DUMP_REG(ACTIVITY2);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act2;
        for (i=0;i<fbpCount*2 && i<8;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY2, _BE0, fbpStatus), regName);
        }
        for (i=0;i<fbpCount*2 && i<8;i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY2_BE0);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GF100 );
        }
    }

    // Per FBP
    DUMP_REG(ACTIVITY3);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY) & act3)
    {
        LwU32 i;
        LwU32 fbpStatus = act3;
        for (i=8;i<fbpCount*2 && i<16;i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY3, _BE8, fbpStatus), regName);
        }
        for (i=8;i<fbpCount*2 && i<16;i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY3_BE8);
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

void grDumpConsolidatedReportGpc_GM200( LwU32 numActiveGpc, LwU32 grIdx )
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

        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_1, GM200 );
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
            PRINT_TPC_REG_Z( gpcId, tpcId, SM_INFO_SUBUNIT_STATUS, GM200 );
        }
    }
}

void grPrintGpuUnitStatus_GM200(LwU32 status, char *regName)
{
    switch (status) {
        case EMPTY:
            DPRINTF_FIELD( regName, QUOTE_ME(EMPTY), status );
            break;
        case ACTIVE:
            DPRINTF_FIELD( regName, QUOTE_ME(ACTIVE), status );
            break;
        case PAUSED:
            DPRINTF_FIELD( regName, QUOTE_ME(PAUSED), status );
            break;
        case QUIESCENT:
            DPRINTF_FIELD( regName, QUOTE_ME(QUIESCENT), status );
            break;
        case PREEMPTED:
            DPRINTF_FIELD( regName, QUOTE_ME(PREEMPTED), status );
            break;
        case STALLED:
            DPRINTF_FIELD( regName, QUOTE_ME(STALLED), status );
            break;
        case FAULTED:
            DPRINTF_FIELD( regName, QUOTE_ME(FAULTED), status );
            break;
        case HALTED:
            DPRINTF_FIELD( regName, QUOTE_ME(HALTED), status );
            break;
        default:
            DPRINTF_FIELD( regName, QUOTE_ME(UNKNOWN), status );
    }  
}

LwU32 grGetSmHwwEsrWarpId_GM200(LwU32 hwwWarpEsr)
{
    return DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _WARP_ID, hwwWarpEsr);
}
