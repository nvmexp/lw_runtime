/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grga100.c
//
//*****************************************************

//
// includes
//
#include "chip.h"
#include "inst.h"
#include "print.h"
#include "ampere/ga100/dev_graphics_nobundle.h"
#include "ampere/ga100/dev_pri_ringmaster.h"
#include "ampere/ga100/dev_pri_ringstation_sys.h"
#include "ampere/ga100/dev_smcarb.h"
#include "ampere/ga100/dev_master.h" // For PMC_BOOT_0
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/hwproject.h"
#include "ampere/ga100/dev_fault.h"
#include "ampere/ga100/hwproject.h"
#include "ampere/ga100/dev_fuse.h"
#include "ampere/ga100/dev_top.h"
#include "ampere/ga100/dev_mmu.h"
#include "ctrl/ctrl2080.h"
#include "fb.h"
#include "gr.h"
#include "g_gr_private.h"  // (rmconfig) implementation prototypes

#define MAX_ASYNC_CE         8
#define MAX_LWDEC            5
#define MAX_GPU_PARTITIONS   8
#define MAX_GPC              8

// SM DFD defines that can't be in the manuals due overlapping (command dependent)
// field definitions
#define LW_PTPC_PRI_SM_DFD_CONTROL_OPCODE                                     31:24 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_OPCODE_AUTO_INC_RD                    0x00000007 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_START_ROW                        7:0 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_END_ROW                         15:8 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER0                          17:16 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER0_NONE                0x00000000 /* RWIVV */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER0_LANE                0x00000001 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER0_ROW                 0x00000002 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER0_SUBP                0x00000003 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER1                          19:18 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER1_NONE                0x00000000 /* RWIVV */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER1_LANE                0x00000001 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER1_ROW                 0x00000002 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER1_SUBP                0x00000003 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER2                          21:20 /* RWIVF */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER2_NONE                0x00000000 /* RWIVV */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER2_LANE                0x00000001 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER2_ROW                 0x00000002 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_AUTO_INC_RD_ITER2_SUBP                0x00000003 /* RW--V */

#define LW_PTPC_PRI_SM_DFD_CONTROL_OPCODE_WR_AUTO_INC_LMT                0x0000000C /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_LANE                   3:0 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_LANE                     7:4 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_OUTSEL                11:8 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_OUTSEL_PC_LOWER 0x00000000 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_OUTSEL_PC_UPPER 0x00000001 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_OUTSEL_WSTATE   0x00000002 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_START_OUTSEL_W_BSTATE 0x00000003 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_OUTSEL                 15:12 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_OUTSEL_PC_LOWER   0x00000000 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_OUTSEL_PC_UPPER   0x00000001 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_OUTSEL_WSTATE     0x00000002 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_OUTSEL_W_BSTATE   0x00000003 /* RW--V */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE0_WE                   16:16 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE1_WE                   17:17 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE2_WE                   18:18 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE3_WE                   19:19 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE4_WE                   20:20 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_LANE5_WE                   21:21 /* RWIVF */        
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_OUTSEL_WE                  23:23 /* RWIVF */        

// SM ARCH values not lwrrently the manuals or provided above by SM DFD
#define LW_PTPC_PRI_SM_ARCH_PARTITION       2  /* SMV Partitions per TPC */
#define LW_PTPC_PRI_SM_ARCH_SUBPARTITION    4  /* Subpartitions per Partition */
#define LW_PTPC_PRI_SM_DFD_CONTROL_WR_AUTO_INC_LMT_END_LANE_MAX           0x00000003 /* RWIVF */  

// Number of data lanes to be read
#define LW_PTPC_PRI_SM_DFD_LANES  4

#define DFD_VAL(f, v)  DRF_NUM(_PTPC, _PRI_SM_DFD_CONTROL, _##f, v)
#define DFD_DEF(f, v)  DRF_DEF(_PTPC, _PRI_SM_DFD_CONTROL, _##f, _##v)

// http://lwbugs/1814624/12
// Set up auto increment read for 4 lanes of data per subpartition

#define SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_0         \
    (DFD_DEF(OPCODE               , AUTO_INC_RD)      | \
     /* Warp 0 to 31 for each subpartition */           \
     DFD_VAL(AUTO_INC_RD_START_ROW, 0          )      | \
     DFD_VAL(AUTO_INC_RD_END_ROW  , 0x0f       )      | \
     DFD_DEF(AUTO_INC_RD_ITER0    , LANE       )      | \
     DFD_DEF(AUTO_INC_RD_ITER1    , SUBP       )      | \
     DFD_DEF(AUTO_INC_RD_ITER2    , ROW        ))

#define SM_DFD_CONTROL_WARP_PC_READ_TPC_EXP             \
     DFD_VAL(AUTO_INC_RD_END_ROW  , 0x0f       )


#define SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_1                     \
    (DFD_DEF(OPCODE                            , WR_AUTO_INC_LMT) | \
     DFD_VAL(WR_AUTO_INC_LMT_START_LANE        , 0              ) | \
     DFD_DEF(WR_AUTO_INC_LMT_END_LANE          , MAX            ) | \
     DFD_DEF(WR_AUTO_INC_LMT_START_OUTSEL      , PC_LOWER       ) | \
     DFD_DEF(WR_AUTO_INC_LMT_END_OUTSEL        , W_BSTATE       ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE0_WE          , 1              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE1_WE          , 1              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE2_WE          , 1              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE3_WE          , 1              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE4_WE          , 0              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_LANE5_WE          , 0              ) | \
     DFD_VAL(WR_AUTO_INC_LMT_OUTSEL_WE         , 1              ))

/*!
 * @brief   Function to see if the offset is in PGRAPH range
 *
 * @param[IN]   regOffset       Register Offset accessed
 *
 * @return  Returns LwBool      LW_FALSE if offset is outside PGRAPH range
 *                              else return LW_TRUE
 */
LwBool
grPgraphOffset_GA100
(
    LwU32 regOffset
)
{
   // If required offset falls inside PGRAPH range
    if (regOffset < DRF_BASE(LW_PGRAPH) || regOffset > DRF_EXTENT(LW_PGRAPH))
        return LW_FALSE;

    return LW_TRUE;
}

/*!
 * @brief   Function to set GrIdx in BAR0 window register to point to
 *          correct PGRAPH
 *
 * @param[IN]   grIdx           GrIdx for which window config is
 *                              requested
 * @param[IN]   bValid          grEngineId set to passed grIdx if TRUE
 *                              else grEngineId set to ILWALID_GR_IDX
 *
 */
void
grConfigBar0Window_GA100
(
    LwU32 grIdx,
    LwBool bvalid
)
{
    if(bvalid)
    {
        grEngineId = grIdx;
        dprintf("Bar0 window configuration successful for GR Engine %d\n", grEngineId);
    }
    else
    {
        grEngineId = ILWALID_GR_IDX;
    }
}

/*!
 * @brief   Function to callwlate the address from the given Register offset
 *
 * @param[IN]   reg           Register offset
 *
 * @return  Returns           Callwlated register value if the PGRAPH offset
 *                            else just return the Register offset
 *
 */
PhysAddr
grCallwlateAddress_GA100
(
    PhysAddr reg
)
{
    PhysAddr regVal;

    if(grEngineId != ILWALID_GR_IDX)
    {
        regVal = (LW_SMC_PRIV_STRIDE * grEngineId) + LW_SMC_PRIV_BASE + (LwU32)reg - DRF_BASE(LW_PGRAPH);
    }
    else
    {
        dprintf("lw: Error: No grIdx provided\n");
        return 0;
    }
    return regVal;
}

/*!
 * @brief   Function to read a 32bit value from a Register offset by using
 *          the TOOLS_WINDOW registers of LwWatch
 *
 * @param[IN]   reg           Register offset
 *
 * @return  Returns           32bit Value read from the register callwlated from the
 *                            Register offset and grIdx
 *
 */
LwU32 grReadReg32_GA100
(
    PhysAddr reg
)
{
    if((pGr[indexGpu].grPgraphOffset( (LwU32) reg )) && pGr[indexGpu].grGetSmcState())
    {
        LwU32 data;
        PhysAddr registerValue;
        LwU32 value;

        // Callwlate address of Register offset reg
        registerValue = pGr[indexGpu].grCallwlateAddress(reg);

        // Write the reg in the TOOLS_WINDOW_ADDRESS register
        osRegWr32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS, (LwU32)registerValue);

        value = osRegRd32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS);

        // Read the data from the TOOLS_WINDOW_DATA register
        data = osRegRd32(LW_PPRIV_SYS_TOOLS_WINDOW_DATA);
        return DRF_VAL(_PPRIV_SYS, _TOOLS_WINDOW_DATA, _DATA, data);
    }
    else
    {
        return grReadReg32_GF100(reg);
    }
}

/*!
 * @brief   Function to write a 32bit value to a Register offset by using
 *          the TOOLS_WINDOW registers of LwWatch
 *
 * @param[IN]   reg           Register offset
 * @param[IN]   data          32bit data to be written
 *
 */
void grWriteReg32_GA100
(
    PhysAddr reg,
    LwU32 data
)
{
    if((pGr[indexGpu].grPgraphOffset( (LwU32) reg )) && pGr[indexGpu].grGetSmcState())
    {
        LwU32 value;
        PhysAddr registerValue;

        // Callwlate address of Register offset reg
        registerValue = pGr[indexGpu].grCallwlateAddress(reg);

        // Write the reg in the TOOLS_WINDOW_ADDRESS register
        osRegWr32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS, (LwU32)registerValue);

        value = osRegRd32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS);

        // Write the data in the TOOLS_WINDOW_DATA register
        osRegWr32(LW_PPRIV_SYS_TOOLS_WINDOW_DATA, data);
    }
    else
    {
        grWriteReg32_GF100(reg, data);
        return;
    }
}

/*!
 * @brief   Function to read a byte from a Register offset by using
 *          the TOOLS_WINDOW registers of LwWatch
 *
 * @param[IN]   reg           Register offset
 *
 * @return  Returns           Byte read from the register callwlated from the
 *                            Register offset and grIdx
 *
 */
LwU8 grReadReg08_GA100
(
    PhysAddr reg
)
{
    if((pGr[indexGpu].grPgraphOffset( (LwU32) reg )) && pGr[indexGpu].grGetSmcState())
    {
        LwU8 data;
        PhysAddr registerValue;
        LwU32 value;

        // Callwlate address of Register offset reg
        registerValue = pGr[indexGpu].grCallwlateAddress(reg);

        // Write the reg in the TOOLS_WINDOW_ADDRESS register
        osRegWr32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS, (LwU32)registerValue);

        value = osRegRd32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS);

        // Read the data from the TOOLS_WINDOW_DATA register
        data = osRegRd08(LW_PPRIV_SYS_TOOLS_WINDOW_DATA);
        return DRF_VAL(_PPRIV_SYS, _TOOLS_WINDOW_DATA, _DATA, data);
    }
    else
    {
        return grReadReg08_GF100(reg);
    }
}

/*!
 * @brief   Function to write a byte to a Register offset by using
 *          the TOOLS_WINDOW registers of LwWatch
 *
 * @param[IN]   reg           Register Offset
 * @param[IN]   data          Byte to be written
 *
 */
void grWriteReg08_GA100
(
    PhysAddr reg,
    LwU8 data
)
{
    if((pGr[indexGpu].grPgraphOffset( (LwU32) reg )) && pGr[indexGpu].grGetSmcState())
    {
        LwU32 value;
        PhysAddr registerValue;

        // Callwlate address of Register offset reg
        registerValue = pGr[indexGpu].grCallwlateAddress(reg);

        // Write the reg in the TOOLS_WINDOW_ADDRESS register
        osRegWr32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS, (LwU32)registerValue);

        value = osRegRd32(LW_PPRIV_SYS_TOOLS_WINDOW_ADDRESS);

        // Write the data in the TOOLS_WINDOW_DATA register
        osRegWr08(LW_PPRIV_SYS_TOOLS_WINDOW_DATA, data);
    }
    else
    {
        grWriteReg08_GF100(reg, data);
        return;
    }
}

/*!
 * @brief   Function to get SMC state. When SMC is enabled
 *          SMC_ARB will allow remapping of GPCs with a specific GrEngine else
 *          all GPCs will be default connected with Syspipe/GrEngine[0]
 *
 * @param[IN]   bEnable         Enable/Disable SMC
 *
 * @return  Returns LW_STATUS
 *                  LW_TRUE is SMC is enabled else LW_FALSE
 */
LwBool
grGetSmcState_GA100
(void)
{
    LwU32 regVal = GPU_REG_RD32(LW_PSMCARB_SYS_PIPE_INFO);

    return FLD_TEST_DRF(_PSMCARB, _SYS_PIPE_INFO, _MODE, _SMC, regVal);
}

/*!
 * @brief   Function to get the number of Gpcs
 *
 * @param[IN]   grIdx          GrIdx for which number of Gpcs is
 *                             requested
 * @return  Returns LwU32      The number of Gpcs
 *
 */
LwU32
grGetNumGpcs_GA100
(
    LwU32 grIdx
)
{
    LwU32 gpcCount = 0;
    LwU32 gpcMask;

    if(LW_FALSE == pGr[indexGpu].grGetSmcState())
    {
        gpcCount = grGetNumGpcs_GF100( grIdx );
    }
    else
    {
        gpcMask = pGr[indexGpu].grGetGpcMask( grIdx );
        //Count the number of 1's in gpcMask to get gpcCount
        if (gpcMask == 0)
            return 0;
        else
        {
            while(gpcMask)
            {
                gpcCount += (gpcMask & 1);
                gpcMask = gpcMask >> 1;
            }
        }
    }
    return gpcCount;
}

/*!
 * @brief   Function to get the mask of Gpcs
 *
 * @param[IN]   grIdx          GrIdx for which GpcMask is
 *                             requested
 * @return  Returns LwU32      GpcMask which indicates which Gpcs are in use
 *
 */
LwU32
grGetGpcMask_GA100
(
    LwU32 grIdx
)
{
    // Read gpcMask
    LwU32 gpcMask = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_1(grIdx));
    return DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_1, _GPC_MASK, gpcMask);
}

/*!
 * @brief   Function to get list of GPC Ids associated with a Gr Engine
 *
 * @param[IN]   grIdx           GR Engine IDx for which mappings are requested
 * @param[OUT]  pGpcId          Array of physical GPC ID
 * @param[OUT]  pGpcCount       GPC Count
 *
 */
void
grGetGpcIdsFromGrIdx_GA100
(
    LwU32       grIdx,
    LwU32       *pGpcPhysId,
    LwU32       *pGpcCount
)
{
    LwU32 i;
    if (LW_FALSE == pGr[indexGpu].grGetSmcState())
    {
        grGetGpcIdsFromGrIdx_GF100(grIdx, pGpcPhysId, pGpcCount);
    }
    else
    {
        LwU32 gpcMask = pGr[indexGpu].grGetGpcMask( grIdx );
        LwU32 numGpcs = pGr[indexGpu].grGetMaxGpc();

        // Return the index and count of all the 1's in the mask
        for (i = 0; i < numGpcs && (gpcMask); i++)
        {
            pGpcPhysId[i] = portUtilCountTrailingZeros32(gpcMask);
            *pGpcCount += 1;
            gpcMask &= ~LWBIT32(pGpcPhysId[i]);
        }
    }
}

/*!
 * @brief   Function to get the number of active Gpcs
 *
 * @param[IN]   grIdx          GrIdx for which number of Gpcs is
 *                             requested
 * @return  Returns LwU32      The number of active Gpcs
 *
 */
LwU32
grGetNumActiveGpc_GA100
(
    LwU32 grIdx
)
{
    LwU32 numActiveGpc;
    if (LW_FALSE == pGr[indexGpu].grGetSmcState())
    {
        numActiveGpc = grGetNumActiveGpc_GK104( grIdx );
    }
    else
    {
        numActiveGpc = pGr[indexGpu].grGetNumGpcs( grIdx );
    }
    return numActiveGpc;
}

/*!
 * @brief   Function to get the max number of Gpcs
 *
 * @return  Returns LwU32      The max number of Gpcs
 *
 */
LwU32 grGetMaxGpc_GA100(void)
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxTpcPerGpc_GA100()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetNumberPesPerGpc_GA100(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}


typedef struct
{
  LwU32 pc_lower;
  LwU32 pc_upper;
  LwU32 run_state;
  LwU32 barrier_state;
} SMDfdWarpData;

static void _grDumpTpcWarpPcs( LwU32 gpcId, LwU32 tpcId, LwU32 repeatPerWarp, LwBool bAnnotate )
{

    // Used by gr.h print macros
    char buffer[GR_REG_NAME_BUFFER_LEN];

    LwU32 warpId, subPartitionId, smWarpId;
    LwU32 warpValidMaskPtr[4];
    LwU32 warpValid = 0;
    LwU32 warpHang = 0;
    LwBool tpcHang = LW_TRUE;

    SMDfdWarpData *warpDataBuffer = NULL;
    LwU32 repeat = 0;
    LwU32 warpCount;

    // Get the number of possible warps
    warpCount = DRF_VAL (_PGRAPH, _PRI_GPC0_TPC0_SM_ARCH, _WARP_COUNT, GPU_REG_RD32 (TPC_REG_ADDR (SM_ARCH, gpcId, tpcId)));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM_ARCH_WARP_COUNT = %u\n", gpcId, tpcId, warpCount);

    if (repeatPerWarp)
    {
        warpDataBuffer = (SMDfdWarpData *) calloc (repeatPerWarp * warpCount, sizeof(SMDfdWarpData));
        if ( warpDataBuffer == NULL )
        {
            dprintf ("Error! Memory could not be allocated.");
            return;
        }
    }

    warpValidMaskPtr [0] = GPU_REG_RD32( SM_REG_ADDR( DBGR_WARP_VALID_MASK_0, gpcId, tpcId, 0));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM0_DBGR_WARP_VALID_MASK_0 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, SM_REG_ADDR( DBGR_WARP_VALID_MASK_0, gpcId, tpcId, 0), warpValidMaskPtr [0]);
    warpValid |= warpValidMaskPtr [0];
    warpValidMaskPtr [1] = GPU_REG_RD32( SM_REG_ADDR( DBGR_WARP_VALID_MASK_1, gpcId, tpcId, 0));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM0_DBGR_WARP_VALID_MASK_1 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, SM_REG_ADDR( DBGR_WARP_VALID_MASK_1, gpcId, tpcId, 0), warpValidMaskPtr [1]);
    warpValid |= warpValidMaskPtr [1];
    warpValidMaskPtr [2] = GPU_REG_RD32( SM_REG_ADDR( DBGR_WARP_VALID_MASK_0, gpcId, tpcId, 1));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM1_DBGR_WARP_VALID_MASK_0 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, SM_REG_ADDR (DBGR_WARP_VALID_MASK_0, gpcId, tpcId, 1), warpValidMaskPtr [2]);
    warpValid |= warpValidMaskPtr [2];
    warpValidMaskPtr [3] = GPU_REG_RD32( SM_REG_ADDR( DBGR_WARP_VALID_MASK_1, gpcId, tpcId, 1));
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM1_DBGR_WARP_VALID_MASK_1 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId, SM_REG_ADDR (DBGR_WARP_VALID_MASK_1, gpcId, tpcId, 1), warpValidMaskPtr [3]);
    warpValid |= warpValidMaskPtr [3];

    if (warpValid)
    {
        if (repeatPerWarp)
        {
            char *inactiveWarp = " (i)";
            LwBool tpcMoving = LW_FALSE;

            dprintf("Warp PCs for GPC %d, TPC %d, (i) == inactive warp\n", gpcId, tpcId );
            dprintf("%12s %12s %7s %4s %18s %5s %10s %13s %6s\n", "SM Warp ID", "Subpartition", "Warp ID", "VEID", "Warp PC", "Valid", "Run State", "Barrier State", "Repeat");

            for ( repeat = 0; repeat < repeatPerWarp; repeat += 1)
            {
                // Turn on the DFD machine and set up the warp data read
                // pattern.  Only print data for active warps, but do the reads
                // for every warp so the DFD auto-increment happens.
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_0);
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_1);

                // [Bug 2198150] Select pipe for reading from SM0
                GPU_REG_WR32(TPC_REG_ADDR(TEX_M_ROUTING, gpcId, tpcId), LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0);

                // Read DFD for SM0
                for ( warpId = 0 ; warpId < warpCount / (2 * LW_PTPC_PRI_SM_ARCH_SUBPARTITION) ; warpId ++ )
                {
                    for (subPartitionId = 0; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION; subPartitionId++)
                    {
                        LwU32 warpDataIndex;
                        smWarpId = warpId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                        warpDataIndex = smWarpId * repeatPerWarp + repeat;
                        warpDataBuffer[warpDataIndex].pc_lower      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].pc_upper      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].run_state     = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].barrier_state = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                    }
                }

                // [Bug 2198150] Select pipe for reading from SM1
                GPU_REG_WR32(TPC_REG_ADDR(TEX_M_ROUTING, gpcId, tpcId), LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1);

                // Read DFD for SM1
                for ( ; warpId < warpCount / LW_PTPC_PRI_SM_ARCH_SUBPARTITION; warpId ++ )
                {
                    for (subPartitionId = 0; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION; subPartitionId++)
                    {
                        LwU32 warpDataIndex;
                        smWarpId = warpId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                        warpDataIndex = smWarpId * repeatPerWarp + repeat;
                        warpDataBuffer[warpDataIndex].pc_lower      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].pc_upper      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].run_state     = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].barrier_state = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                    }
                }

                // Shut down DFD machine
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), 0x0);
            }

            for ( warpId = 0 ; warpId < (warpCount / LW_PTPC_PRI_SM_ARCH_SUBPARTITION) ; warpId ++ )
            {
                for ( subPartitionId = 0 ; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION ; subPartitionId ++ )
                {
                    LwU64 lastPC = 0x0;
                    LwU32 lastRunState = 0x0;
                    LwU32 lastBarrierState = 0x0;
                    LwU32 pcRepeatCount = 0;
                    // Volta+ has 2 SMs per TPC. The arch number
                    // is per TPC, so divide by 2 for per-SM warp
                    // count.
                    LwU32 smWarpCount = warpCount / 2;
                    LwBool warpIsValid;
                    smWarpId = warpId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                    dprintf("%12d %12d %7d", smWarpId % smWarpCount, subPartitionId, warpId);

                    for ( repeat = 0; repeat < repeatPerWarp; repeat += 1 )
                    {
                        LwU32 veId, warpDataIndex;
                        char *warpValidStr;
                        LwU64 warpPC;
                        LwU32 runState;
                        LwU32 barrierState;

                        warpDataIndex = smWarpId * repeatPerWarp + repeat;
                        warpPC = warpDataBuffer[warpDataIndex].pc_upper & 0x0001FFFF;
                        warpPC = (warpPC << 32) | warpDataBuffer[warpDataIndex].pc_lower;
                        veId = (warpDataBuffer[warpDataIndex].pc_upper >> 24) & 0xFF;
                        warpIsValid = warpPC != 0x1;
                        warpValidStr = warpIsValid ? "" : inactiveWarp;
                        runState = warpDataBuffer[warpDataIndex].run_state;
                        barrierState = warpDataBuffer[warpDataIndex].barrier_state;

                        if (lastPC == warpPC && lastRunState == runState && lastBarrierState == barrierState) {
                            pcRepeatCount += 1;
                        }
                        else {
                            if (pcRepeatCount) {
                                dprintf(" x %4d", pcRepeatCount + 1);
                            }

                            pcRepeatCount = 0;

                            if (repeat) {
                                dprintf("\n%33s", " ");
                            }

                            dprintf(" %4d 0x%.16llx %5s 0x%.8x    0x%.8x", veId, warpPC, warpValidStr, runState, barrierState);
                            lastPC = warpPC;
                            lastRunState = runState;
                            lastBarrierState = barrierState;
                        }
                    }

                    // Print the repeat count for the last set of samples
                    if (pcRepeatCount) {
                        dprintf(" x %4d", pcRepeatCount + 1);
                    }
                    dprintf("\n");

                    if (pcRepeatCount + 1 == repeatPerWarp && warpIsValid) {
                        warpHang += 1;
                    } else {
                        tpcHang = LW_FALSE;
                    }
                }
            }
        }

        PRINT_SM_REG_PD ( DBGR_BPT_PAUSE_MASK_0, gpcId, tpcId, 0 );
        PRINT_SM_REG_PD ( DBGR_BPT_PAUSE_MASK_1, gpcId, tpcId, 0 );
        PRINT_SM_REG_PD ( DBGR_BPT_PAUSE_MASK_0, gpcId, tpcId, 1 );
        PRINT_SM_REG_PD ( DBGR_BPT_PAUSE_MASK_1, gpcId, tpcId, 1 );

        PRINT_SM_REG_PD ( DBGR_BPT_TRAP_MASK_0, gpcId, tpcId, 0 );
        PRINT_SM_REG_PD ( DBGR_BPT_TRAP_MASK_1, gpcId, tpcId, 0 );
        PRINT_SM_REG_PD ( DBGR_BPT_TRAP_MASK_0, gpcId, tpcId, 1 );
        PRINT_SM_REG_PD ( DBGR_BPT_TRAP_MASK_1, gpcId, tpcId, 1 );

        if (repeat > 8 && warpHang > 0) {
            dprintf("GPC %d TPC %d: %d / %d warps PCs not moving with %d samples. TPC appears %shang\n",
                    gpcId, tpcId, warpHang, warpCount, repeat, tpcHang ? "" : "not ");
        }
    }

    if (warpDataBuffer)
    {
        free (warpDataBuffer);
    }
}

void grDumpWarpPc_GA100(LwU32 gpcId, LwU32 tpcId, LwU32 repeat, LwBool bAnnotate)
{
    _grDumpTpcWarpPcs(gpcId, tpcId, repeat, bAnnotate);
}

/*!
 * @brief   Function to get SMC partition info specific to given swizzId(if provided)
 *          or to get SMC partition info of all configured partitions
 *
 * @param[IN]   swizzId        SwizzId for which partition info is
 *                             requested. If swizzId is invalid, info of all
 *                             partitions is requested
 *
 */
void
grGetSmcPartitionInfo_GA100
(
    LwU32 swizzId
)
{
    LwU32 i, j;
    LwU32 swizzIdCount = 0;
    LwU32 remoteSwizzId;
    LwBool bSwizzIdAbsent = LW_FALSE;
    LwU32 pSwizzIdsUsed[MAX_SWIZZID];

    if(swizzId == ILWALID_SWIZZID)
    {
        //
        // When partition information of the complete GPU has been requested
        // we loop over all the swizzIds that are lwrretly enabled in the system
        // and print information corresponding to each enabled swizzId.
        //
        for(i = 0; i < MAX_GR_IDX; i++)
        {
            bSwizzIdAbsent = LW_TRUE;
            remoteSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(i));
            remoteSwizzId = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _REMOTE_SWIZID, remoteSwizzId);

            if(remoteSwizzId != 0)
            {
                // Check to see if swizzId is already present in the pSwizzIdsUsed array
                for(j = 0; j < swizzIdCount; j++)
                {
                    if(pSwizzIdsUsed[j] == remoteSwizzId)
                    {
                        bSwizzIdAbsent = LW_FALSE;
                        break;
                    }
                }
                // Add each swizzId to the pSwizzIdsUsed array only once
                if(bSwizzIdAbsent)
                {
                    pSwizzIdsUsed[swizzIdCount++] = remoteSwizzId;
                }
            }
        }
    }
    else
    {
        //
        // When a specific swizzId is passed as a parameter, first we check if 
        // any partition with that swizzId exists and only then information
        // specific to the swizzId is displayed
        //
        for( i = 0; i < MAX_GR_IDX; i++)
        {
            remoteSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(i));
            remoteSwizzId = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _REMOTE_SWIZID, remoteSwizzId);

            if( remoteSwizzId == swizzId )
            {
                swizzIdCount = 1;
                pSwizzIdsUsed[0] = swizzId;
                break;
            }
        }
        if(swizzIdCount == 0)
        {
            dprintf("lw: Error: No partitions available with swizzId %d\n", swizzId);
            return;
        }
    }

    for(i = 0; i < swizzIdCount; i++)
    {
        pGr[indexGpu].grPrintPartitionInfo(pSwizzIdsUsed[i]);
    }
}

/*!
 * @brief   Function to print SMC partition info specific to given swizzId
 *
 * @param[IN]   swizzId        SwizzId for which partition info is
 *                             requested
 *
 */
void
grPrintPartitionInfo_GA100
(
    LwU32 swizzId
)
{
    LwU32 partitionTableMask;
    LwU32 pGrIndiciesUsed[MAX_GR_IDX];
    LwU32 sysPipeMask = 0;
    LwU32 grEngCount = 0;
    LwU64 partitionStartAddr = 0;
    LwU64 partitionEndAddr = 0;
    LwU64 partitionSizeInBytes = 0;
    LwU64 partitionableMemStartAddr = 0;
    LwU64 partitionableMemEndAddr = 0;
    LwU32 gpcCountTotal = 0;
    LwU32 veidOffsetTotal = 0;
    LwU32 veidCountTotal = 0;
    LwU32 veidSmcMapTotal = 0;
    LwU32 gpcCountPerGr = 0;
    LwU32 veidOffsetPerGr = 0;
    LwU32 veidCountPerGr = 0;
    LwU32 veidSmcMapPerGr = 0;
    LwU32 ceCount = 0;
    LwU32 lwDecCount = 0;
    LwU32 k;
    LW_STATUS status;

    partitionTableMask = pGr[indexGpu].grGetPartitionTableMask(swizzId);

    pGr[indexGpu].grGetSysPipeMask(swizzId, &sysPipeMask, &grEngCount, pGrIndiciesUsed);

    status = pGr[indexGpu].grSwizzIdToEngineConfig(swizzId, &lwDecCount, &ceCount);

    if(status != LW_OK)
    {
        dprintf("lw: Error: Error getting the engine Config for swizzId %d\n", swizzId);
        return;
    }

    for(k = 0; k < grEngCount; k++)
    {
        gpcCountPerGr = pGr[indexGpu].grGetNumGpcs(pGrIndiciesUsed[k]);
        gpcCountTotal += gpcCountPerGr;

        pGr[indexGpu].grGetVeidSmcConfig(pGrIndiciesUsed[k], &veidOffsetPerGr, &veidCountPerGr, &veidSmcMapPerGr);

        if( k == 0)
        {
            veidOffsetTotal = veidOffsetPerGr;
        }
        veidCountTotal += veidCountPerGr;
        veidSmcMapTotal |= veidSmcMapPerGr;
    }

    dprintf("------------------------------------------------\n");
    dprintf("| SwizzId  | Partition Table Mask  |  GPC Count|\n");
    dprintf("------------------------------------------------\n");
    dprintf("|    %2d    |      0x%3x            |     %1d     |\n", swizzId, partitionTableMask, gpcCountTotal);
    dprintf("---------------------------------------------------\n");
    dprintf("| GrEngine Count   |  CE Count  |  LWDEC Count    |\n");
    dprintf("---------------------------------------------------\n");
    dprintf("|    %1d             |       %1d    |      %1d          |\n",
              grEngCount, ceCount, lwDecCount);
    dprintf("-----------------------------------------------------\n");
    dprintf("| VEID Offset    |   VEID Count    | VEID-SMC Map   |\n");
    dprintf("-----------------------------------------------------\n");
    dprintf("|     %2d         |       %2d        | 0x%8x     |\n", veidOffsetTotal, veidCountTotal, veidSmcMapTotal);

    pGr[indexGpu].grGetPartitionMemInfo(swizzId, &partitionableMemStartAddr, &partitionableMemEndAddr,
                                        &partitionStartAddr, &partitionEndAddr, &partitionSizeInBytes);

    dprintf("-----------------------------------------------------\n");
    dprintf("| Global Partitionable   |   Global Partitionable  |\n");
    dprintf("|  Memory Start Addr     |    Memory End Addr      |\n");
    dprintf("----------------------------------------------------\n");
    dprintf("|       0x%8llx       |   0x%8llx            |\n",
              partitionableMemStartAddr, partitionableMemEndAddr);
    dprintf("----------------------------------------------------\n");
    dprintf("------------------------------------------------------------------------\n");
    dprintf("| Local Partition        |    Local Partition      |  Local Partition  |\n");
    dprintf("| Memory Start Addr      |    Memory End Addr      |  Size In Bytes    |\n");
    dprintf("------------------------------------------------------------------------\n");
    dprintf("|       0x%8llx       |   0x%8llx            |   %10lld      |\n",
               partitionStartAddr, partitionEndAddr, partitionSizeInBytes);
    dprintf("------------------------------------------------------------------------\n");
    dprintf("=======================================================================================\n");
}

/*!
 * @brief   Function to get SMC partition info specific to given grIdx
 *
 * returns Returns partition information for a specific GR Engine Idx
 */
void
grGetSmcEngineInfo_GA100
(void)
{
    LwU32 partitionTableMask;
    LwU32 swizzId;
    LwU32 veidCount, veidSmcMap, veidOffset;
    LwU32 gpcCount;
    LwU32 gpcSmcMap[MAX_GPC];
    LW_STATUS status;

    status = pGr[indexGpu].grGetSwizzIdFromGrIdx(grEngineId, &swizzId);
    if(status != LW_OK)
    {
        dprintf("lw: Error: Unable to get a swizzId for GR Engine %d\n", grEngineId);
        return;
    }

    partitionTableMask = pGr[indexGpu].grGetPartitionTableMask(swizzId);
    gpcCount = pGr[indexGpu].grGetNumGpcs(grEngineId);

    dprintf("-------------------------------------------------\n");
    dprintf("| SwizzId  | Partition Table Mask  |  GPC Count |\n");
    dprintf("------------------------------------------------\n");
    dprintf("|    %2d    |      0x%3x            |     %1d     |\n", swizzId, partitionTableMask, gpcCount);
    dprintf("-------------------------------------------------\n");

    pGr[indexGpu].grGetGpcSmcMap(grEngineId, gpcSmcMap);
    pGr[indexGpu].grGetVeidSmcConfig(grEngineId, &veidOffset, &veidCount, &veidSmcMap);

    dprintf("-----------------------------------------------------\n");
    dprintf("| VEID Offset    |   VEID Count    | VEID-SMC Map   |\n");
    dprintf("-----------------------------------------------------\n");
    dprintf("|     %2d         |       %2d        | 0x%8x     |\n", veidOffset, veidCount, veidSmcMap);
    dprintf("-----------------------------------------------------\n");
}

/*!
 * @brief   Function to get swizzId corresponding to the given grIdx
 *
 * @param[IN]   grIdx          GrIdx for which swizzId is requested
 *
 * @return  Returns LW_STATUS
 *                  LW_OK  
 *                  LW_ERR_ILWALID_ARGUMENT if swizzId of GR Index is 0 
 *                                          when no other Gr engines have 
 *                                          swizzid 0  
 *
 */
LW_STATUS
grGetSwizzIdFromGrIdx_GA100
(
    LwU32 grIdx,
    LwU32 *pSwizzId
)
{
    LwU32 i;
    LwU32 swizzId;
    LwU32 remoteSwizzId;
    swizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(grIdx));
    *pSwizzId = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _REMOTE_SWIZID, swizzId);

    if(*pSwizzId == 0)
    {
        for(i = 0; i < MAX_GR_IDX; i++)
        {
            remoteSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(i));
            remoteSwizzId = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _REMOTE_SWIZID, remoteSwizzId);

            if(remoteSwizzId != 0)
            {
                dprintf("lw: Error: GR Engine %d is not associated with any swizzId\n", grIdx);
                return LW_ERR_ILWALID_ARGUMENT;                
            }
        }
    }
    return LW_OK;
}

/*!
 * @brief   Function to get the syspipe mask in a partition
 *
 * @param[IN]   swizzId           SwizzlId used by partition
 *
 * @param[OUT]  pSysPipeMask      Syspipes/Gr engines Mask in given partition
 * @param[OUT]  pGrEngCount       Number of syspipes/gr engines in given partition
 * @param[OUT]  pGrIndiciesUsed   Gr engine Ids used in given partition
 *
 */
void
grGetSysPipeMask_GA100
(
    LwU32 swizzId,
    LwU32 *pSysPipeMask,
    LwU32 *pGrEngCount,
    LwU32 *pGrIndiciesUsed
)
{
    LwU32 i;
    LwU32 j = 0;
    LwU32 remoteSwizzId;
    *pSysPipeMask = 0x0;
    *pGrEngCount = 0;

    for( i = 0; i < MAX_GR_IDX; i++)
    {
        remoteSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(i));
        remoteSwizzId = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _REMOTE_SWIZID, remoteSwizzId);

        if( remoteSwizzId == swizzId )
        {
            *pSysPipeMask |= LWBIT32(i);
            dprintf("--------------------------\n");
            dprintf("|Swizz ID %2d -> GR ENG %d |\n", swizzId, i);
            dprintf("--------------------------\n");
            pGrIndiciesUsed[j] = i;
            *pGrEngCount += 1;
            j++;
        }
    }
}

/*!
 * @brief   Function to get physical CE IDs/count and physical
 *          LWDEC IDs/count in a partition
 *
 * @param[IN]   swizzId       SwizzId used by partition
 *
 * @param[OUT]  pLwDecCount   Number of LWDECs in given partition
 * @param[OUT]  pCeCount      Number of CEs in given partition
 *
 * @returnReturns LW_STATUS
 *                LW_OK       if deviceinfo is obtained else ERROR
 *
 */
LW_STATUS
grSwizzIdToEngineConfig_GA100
(
    LwU32 swizzId,
    LwU32 *pLwDecCount,
    LwU32 *pCeCount
)
{
    LwU32 i;
    LwU32 ceFaultIndex = 0;
    LwU32 lwDecFaultIndex = 0;
    LwU32 remoteEngSwizzId;
    LwU32 lwDecIds[MAX_LWDEC];
    LwU32 ceIds[MAX_ASYNC_CE];
    LwU32 mmuCeFaultId[MAX_ASYNC_CE];
    LwU32 mmuLwDecFaultId[MAX_LWDEC];
    LW_STATUS status;

    *pCeCount = 0;
    *pLwDecCount = 0;

    status = pFifo[indexGpu].fifoGetDeviceInfo();
    if(status != LW_OK)
    {
        return status;
    }

    for(i = 0; i < deviceInfo.enginesCount; i++)
    {
        if(deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENGINE_TAG] == ENGINE_TAG_CE)
        {
            mmuCeFaultId[ceFaultIndex] = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_MMU_FAULT_ID];
            remoteEngSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_MMU_ENG_ID_CFG(mmuCeFaultId[ceFaultIndex]));
            remoteEngSwizzId = DRF_VAL(_PFB, _PRI_MMU_MMU_ENG_ID_CFG, _REMOTE_SWIZID, remoteEngSwizzId);

            if(remoteEngSwizzId == swizzId)
            {
                ceIds[*pCeCount] = mmuCeFaultId[ceFaultIndex] - mmuCeFaultId[0];
                dprintf("| CE Ids = %2d    |\n", ceIds[*pCeCount]);
                *pCeCount += 1;
            }
            ceFaultIndex++;
        }
        else if(deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENGINE_TAG] == ENGINE_TAG_LWDEC)
        {
            mmuLwDecFaultId[lwDecFaultIndex] = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_MMU_FAULT_ID];
            remoteEngSwizzId = GPU_REG_RD32(LW_PFB_PRI_MMU_MMU_ENG_ID_CFG(mmuLwDecFaultId[lwDecFaultIndex]));
            remoteEngSwizzId = DRF_VAL(_PFB, _PRI_MMU_MMU_ENG_ID_CFG, _REMOTE_SWIZID, remoteEngSwizzId);

            if(remoteEngSwizzId == swizzId)
            {
                lwDecIds[*pLwDecCount] = mmuLwDecFaultId[lwDecFaultIndex] - mmuLwDecFaultId[0];
                dprintf("| LWDEC Ids = %d  |\n", lwDecIds[*pLwDecCount]);
                *pLwDecCount += 1;
            }
            lwDecFaultIndex++;
        }
        else
        {
            continue;
        }
    }
    return LW_OK;
}

/*!
 * @brief   Function to get partition table mask which need to be set for a
 *          specific swizzID in partition table
 *
 * @param[IN]   swizzId         SwizzId used by partition
 *
 * @return  Returns LwU32 mask for partition Table
 *
 */
LwU32
grGetPartitionTableMask_GA100
(
    LwU32 swizzId
)
{
    switch ( swizzId )
    {
        case 0:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_0;

        case 1:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_1;

        case 2:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_2;

        case 3:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_3;

        case 4:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_4;

        case 5:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_5;

        case 6:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_6;

        case 7:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_7;

        case 8:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_8;

        case 9:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_9;

        case 10:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_10;

        case 11:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_11;

        case 12:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_12;

        case 13:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_13;

        case 14:
            return LW_PFB_FBHUB_MEM_PARTITION_SWIZID_MASK_14;

        default:
            {
                dprintf("lw: Error: Invalid swizzId\n");
                return 0;
            }
    }
}

/*!
 * @brief   Function to get HW GPC to SMC mappings.
 *
 * @param[IN]   grIdx           GR Engine Idx
 *
 * @param[OUT]  pGpcSmcMap      Mapping of physical GPC Ids connected
 *                              to the GrIdx
 */
void
grGetGpcSmcMap_GA100
(
   LwU32 grIdx,
   LwU32 *pGpcSmcMap
)
{
    LwU32 i;
    LwU32 sysPipeId;
    LwU32 gpcPhysId[MAX_GPC];
    LwU32 gpcCount = 0;

    pGr[indexGpu].grGetGpcIdsFromGrIdx(grIdx, gpcPhysId, &gpcCount);

    if(gpcCount == 0)
    {
        dprintf("lw: Error: No GPCs configured\n");
        return;
    }

    for( i = 0; i < gpcCount; i++ )
    {
        sysPipeId = GPU_REG_RD32(LW_PSMCARB_SMC_PARTITION_GPC_MAP(gpcPhysId[i]));
        sysPipeId = DRF_VAL(_PSMCARB, _SMC_PARTITION_GPC_MAP, _SYS_PIPE_ID, sysPipeId);

        pGpcSmcMap[gpcPhysId[i]] = sysPipeId;
        dprintf("-------------------\n");
        dprintf("|GR-ID %d -> GPC %d |\n", pGpcSmcMap[gpcPhysId[i]], gpcPhysId[i]);
        dprintf("-------------------\n");
    }
}

/*!
 * @brief   Function to configure VEIDs for GrEngines
 *
 * @param[IN]   grIdx          GR Engine Idx
 *
 * @param[OUT]   pVeidOffset    Veid Offset of given grIdx
 * @param[OUT]   pVeidCount     Veid count of given grIdx
 * @param[OUT]   pVeidSmcMap    Mapping of Veids to given grIdx
 *
 */
void
grGetVeidSmcConfig_GA100
(
    LwU32 grIdx,
    LwU32 *pVeidOffset,
    LwU32 *pVeidCount,
    LwU32 *pVeidSmcMap
)
{
    LwU32 numGpcs = pGr[indexGpu].grGetNumGpcs( grIdx );

    *pVeidOffset = GPU_REG_RD32(LW_PFB_PRI_MMU_SMC_ENG_CFG_0(grIdx));
    *pVeidOffset = DRF_VAL(_PFB, _PRI_MMU_SMC_ENG_CFG_0, _MMU_ENG_VEID_OFFSET, *pVeidOffset);

    *pVeidCount = 8 * numGpcs;

    *pVeidSmcMap = DRF_MASK64(*pVeidCount - 1:0) << (*pVeidOffset);
}

/*!
 * @brief   Function to map swizzId to supported memory size and addresses
 *
 * @param[IN]   swizzId                     SwizzleId used by partition
 *
 * @param[OUT]  pPartitionableMemStartAddr   Start address of partitionable memory
 * @param[OUT]  pPartitionableMemEndAddr     End address of partitionable memory
 * @param[OUT]  pStartAddr                  Starting address of memory
 * @param[OUT]  pEndAddr                    End address of memory
 * @param[OUT]  pSizeInBytes                Memory size in bytes supported by partition
 *
 * @return  Returns LW_STATUS
 *          LW_OK
 *          LW_ERR_ILWALID_ARGUMENT   If un-supported partition size is
 *                                    requested
 */
LW_STATUS
grGetPartitionMemInfo_GA100
(
    LwU32 swizzId,
    LwU64 *pPartitionableMemStartAddr,
    LwU64 *pPartitionableMemEndAddr,
    LwU64 *pStartAddr,
    LwU64 *pEndAddr,
    LwU64 *pSizeInBytes
)
{
    LW_STATUS status;
    LwU64 unalignedStartAddr;
    LwU32 partitionSizeFlag = 0;
    LwU32 minSwizzId = 0;
    LwU64 vmmuSegmentSize = 0;

    LwU32   regValMiddle = GPU_REG_RD32(LW_PFB_PRI_MMU_MEM_PARTITION_MIDDLE);
    LwU32   regValBottom = GPU_REG_RD32(LW_PFB_PRI_MMU_MEM_PARTITION_BOTTOM);

    LwU32   numActiveLTCs = pFb[indexGpu].fbGetActiveLTCCountLwW();
    LwU32   numLTSPerLTC  = pFb[indexGpu].fbGetLTSPerLTCCountLwW();

    *pPartitionableMemStartAddr = regValBottom * numActiveLTCs * numLTSPerLTC * LW_PFB_PRI_MMU_MEM_PARTITION_ATOM_SIZE;
    *pPartitionableMemEndAddr = (regValMiddle * (numActiveLTCs * numLTSPerLTC * LW_PFB_PRI_MMU_MEM_PARTITION_ATOM_SIZE)) - 1;

    status = pGr[indexGpu].grGetPartitionMemSize(swizzId, pPartitionableMemStartAddr, pPartitionableMemEndAddr,
                                                 &minSwizzId, &partitionSizeFlag, pSizeInBytes);
    if(status != LW_OK)
        return status;

    // Memory addresses need to be VMMU segment size aligned.
    vmmuSegmentSize = pMmu[indexGpu].mmuGetSegmentSize();

    unalignedStartAddr = (*pPartitionableMemStartAddr + (*pSizeInBytes * (swizzId - minSwizzId)));

    *pStartAddr = LW_ALIGN_UP64(unalignedStartAddr, vmmuSegmentSize);
    *pEndAddr = LW_ALIGN_DOWN64((unalignedStartAddr + *pSizeInBytes), vmmuSegmentSize) - 1;
    *pSizeInBytes = *pEndAddr - *pStartAddr + 1;

    return LW_OK;
}

/*!
 * @brief   Function to map swizzId to supported partition memory size
 *
 * @param[IN]   swizzId             SwizzleId used by partition
 * @param[IN]   pPartitionStartAddr Partitionable Memory Start Address
 * @param[IN]   pPartitionEndAddr   Partitionable Memory End Address
 *
 * @param[OUT]  pMinswizzId         Minimum swizzId corresponding to partitonSizeFlag
 * @param[OUT]  pPartitionSizeFlag  Flag stating partition memory size
 * @param[OUT]  pSizeInBytes        Memory size in bytes supported by partition
 *
 * @return  Returns LW_STATUS
 *          LW_OK
 *          LW_ERR_ILWALID_ARGUMENT         If un-supported swizzId
 *          LW_ERR_INSUFFICIENT_RESOURCES   If there isn't enough memory for partiton
 *                                          to be allocated
 */
LW_STATUS
grGetPartitionMemSize_GA100
(
    LwU32       swizzId,
    LwU64       *pPartitionStartAddr,
    LwU64       *pPartitionEndAddr,
    LwU32       *pMinSwizzId,
    LwU32       *pPartitionSizeFlag,
    LwU64       *pSizeInBytes
)
{
    //
    // To handle the straddling issue we always consider memory for different
    // swizzIds as addition of minimum sized segements allowed in partitioning
    //
    LwU64 memSize = (*pPartitionEndAddr - *pPartitionStartAddr + 1) /
                     MAX_GPU_PARTITIONS;

    switch (swizzId)
    {
        case 0:
        {
            *pSizeInBytes = memSize * MAX_GPU_PARTITIONS;
            *pPartitionSizeFlag = LW2080_CTRL_GPU_PARTITION_FLAG_FULL_GPU;
            *pMinSwizzId = 0;
            break;
        }

        case 1:
        case 2:
        {
            *pSizeInBytes = (memSize * (MAX_GPU_PARTITIONS / 2));
            *pPartitionSizeFlag = LW2080_CTRL_GPU_PARTITION_FLAG_ONE_HALF_GPU;
            *pMinSwizzId = 1;
            break;
        }

        case 3:
        case 4:
        case 5:
        case 6:
        {
            *pSizeInBytes = (memSize * (MAX_GPU_PARTITIONS / 4));
            *pPartitionSizeFlag = LW2080_CTRL_GPU_PARTITION_FLAG_ONE_QUARTER_GPU;
            *pMinSwizzId = 3;
            break;
        }

        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        {
            *pSizeInBytes = memSize;
            *pPartitionSizeFlag = LW2080_CTRL_GPU_PARTITION_FLAG_ONE_EIGHTHED_GPU;
            *pMinSwizzId = 7;
            break;
        }

        default:
        {
            dprintf("lw: Error: Unsupported SwizzId\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }
    }

    if (*pSizeInBytes == 0)
    {
        dprintf("lw: Error: Not enough memory to allocate for partition\n");
        return LW_ERR_INSUFFICIENT_RESOURCES;
    }
    return LW_OK;
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
LwBool grGetActiveGpcConfig_GA100(LwU32 *activeGpcConfig, LwU32 *maxNumberOfGpcs)
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
LwBool grGetActiveTpcConfig_GA100(LwU32 GpcId, LwU32 *activeTpcConfig, LwU32 *maxNumberOfTpcs)
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
LwBool grGetActiveZlwllConfig_GA100(LwU32 GpcId, LwU32 *activeZlwllConfig, LwU32 *maxNumberOfZlwlls)
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
LwBool grGetActiveFbpConfig_GA100(LwU32 *activeFbpConfig, LwU32 *maxNumberOfFbps)
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
LwBool grGetActiveFbpaConfig_GA100(LwU32 *activeFbpaConfig, LwU32 *maxNumberOfFbpas)
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
LwBool grGetActiveLwencConfig_GA100(LwU32 *activeLwencConfig, LwU32 *maxNumberOfLwencs)
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

// Gets active DISP_HEAD config.
LwBool grGetActiveDispHeadConfig_GA100(LwU32 *activeDispHeadConfig, LwU32 *maxNumberOfDispHeads)
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

void grDumpBeActivity_GA100
(
    LwU32 grStatus,
    LwU32 fbpCount
)
{
    char regName[GR_REG_NAME_BUFFER_LEN];
    char buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
    LwU32 act3 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY3);
    LwU32 act8 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY8);
    LwU32 val;
    
    // Per FBP
    DUMP_REG(ACTIVITY2);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act2;
        for (i = 0 ; i < fbpCount*2 && i < 8 ; i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY2, _BE0, fbpStatus), regName);
        }
        for (i = 0 ; i < fbpCount*2 && i < 8 ; i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY2_BE0);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
        }
    }

    // Per FBP
    DUMP_REG(ACTIVITY3);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act3;
        for (i = 8; i < fbpCount*2 && i < 16; i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY3, _BE8, fbpStatus), regName);
        }
        for (i = 8; i < fbpCount*2 && i < 16; i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY3_BE8);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
        }
    }

    DUMP_REG(ACTIVITY8);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        LwU32 i;
        LwU32 fbpStatus = act3;
        for (i = 16; i < fbpCount*2 && i < 24; i++)
        {
            sprintf(regName, "BE%d", i);
            pGr[indexGpu].grPrintGpuUnitStatus( DRF_VAL( _PGRAPH, _ACTIVITY3, _BE8, fbpStatus), regName);
        }
        for (i = 16; i < fbpCount*2 && i < 24; i++)
        {
            fbpStatus >>= DRF_SIZE(LW_PGRAPH_ACTIVITY3_BE8);
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
        }
    }

}

/*----------------------------------------------------------------------------
 * void
 * grDumpConsolidatedReport_GA100()
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
void grDumpConsolidatedReport_GA100( LwU32 grIdx )
{
    LwU32 grStatus, fbpCount, gpcCount, val;
    LwU32   act0, act1, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];
    
    //print SMCARB from _PGRAPH_STATUS starting from Ampere
    PRINT_REG2_Z(_PGRAPH, _STATUS, GA100);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GV100 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GV100 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GA100 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GV100 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    grStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act4 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY4);

    // For each unit, if it'd not IDLE in LW_PGRAPH_STATUS, print the
    // associated activity value from LW_PGRAPH_ACTIVITY*

    DUMP_REG(ACTIVITY0);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PD, act0),
                        "PD");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _PDB, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _PDB, act0),
                        "PDB");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SCC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SCC, act0),
                        "SCC");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _RASTWOD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _RASTWOD, act0),
                        "RASTWOD");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SSYNC, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SSYNC, act0),
                        "SSYNC");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _CWD, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _CWD, act0),
                        "CWD");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SKED, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SKED, act0),
                        "SKED");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SMCARB, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY0, _SMCARB, act0),
                        "SMCARB");
    }


    DUMP_REG(ACTIVITY1);


    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _MEMFMT, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _MEMFMT, act1),
                        "MEMFMT");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SEMAPHORE, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _SEMAPHORE, act1),
                        "SEMAPHORE");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_FUNNEL, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FUNNEL, act1),
                        "FE_FUNNEL");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_CONST, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _FECONST, act1),
                        "FECONST");
    }
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _TPC_MGR, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _TPCMGR, act1),
                        "TPCMGR");
    }


    pGr[indexGpu].grDumpBeActivity(grStatus, fbpCount);

    // Per GPC
    DUMP_REG(ACTIVITY4);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _GPC, _BUSY))
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

/*----------------------------------------------------------------------------
 * Get count of total number of mailbox register for a given chip from dev_graphics_nobundle.h
 * and print all the mailbox registers
 *----------------------------------------------------------------------------
 */


void grDumpFecsMailboxRegisters_GA100(void)
{
    LwU32 regMbCnt;
    char    buffer[GR_REG_NAME_BUFFER_LEN];
    dprintf("\n==========\nFECS MAILBOX registers\n==========\n");
    for(regMbCnt=0; regMbCnt< LW_PGRAPH_PRI_FECS_CTXSW_MAILBOX__SIZE_1; regMbCnt++)
    {
        sprintf( buffer, "LW_PGRAPH_PRI_FECS_CTXSW_MAILBOX(%d)",regMbCnt);
        priv_dump(buffer);
    }
    for(regMbCnt=0; regMbCnt< LW_PGRAPH_PRI_FECS_CTXSW_FUNCTION_TRACING_MAILBOX__SIZE_1; regMbCnt++)
    {
        sprintf( buffer, "LW_PGRAPH_PRI_FECS_CTXSW_FUNCTION_TRACING_MAILBOX(%d)",regMbCnt);
        priv_dump(buffer);
    }
}

//SMCARB
void grDumpSmcarbRegisters_GA100(void)
{
    dprintf("\n==========\nSMCARB registers\n==========\n");

    PRINT_REG_PD(_PSMCARB, _DEBUG);
    PRINT_REG_PD(_PSMCARB, _RELEASE_STATUS);
}

//compared to function in grgp100.c, remove grPrintEngineGraphicsStatus() as LW_PFIFO_ENGINE_STATUS is removed in AMPERE. See grPrintEngineGraphicsStatus define in GK104
void grDumpDetailedStatus_GA100( BOOL bFullPrint, LwU32 grIdx )
{
    LwU32 numActiveFbp, fbpId;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("********************************************************************************\n");
    dprintf("====================\n");
    dprintf("Detailed GPU status.\n");
    dprintf("====================\n\n");

    // The same high level info as in the consolidated report, but
    // print all registers/fields even if 0
    PRINT_REG_PD(_PGRAPH_GRFIFO, _STATUS);
    PRINT_REG_PD(_PGRAPH_GRFIFO, _CONTROL);

    PRINT_REG_PD(_PGRAPH, _STATUS);
    PRINT_REG_PD(_PGRAPH, _STATUS1);
    PRINT_REG_PD(_PGRAPH, _STATUS2);
    PRINT_REG_PD(_PGRAPH, _INTR);
    PRINT_REG_PD(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );

    // LW_PGRAPH_ACTIVITY*
    PRINT_REG_PD(_PGRAPH, _ACTIVITY*);

    // program LW_PGRAPH_PRI_FE_PERFMON register to enable the perfmon reporting
    pGr[indexGpu].grEnableFePerfmonReporting();

    pGr[indexGpu].grPrintExceptionStatusRegister();

    // Full pipeline status - large printout
    // FECS and GPCCS detail is reported in grCheckPipelineStatus
    pGr[indexGpu].grCheckPipelineStatus( NULL, bFullPrint, grIdx );

    PRINT_REG_PD(_PGRAPH_FECS, _INTR);
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _HOST_INT_STATUS);

    PRINT_REG_PD(_PGRAPH_PRI_DS, _MPIPE_STATUS);
    PRINT_REG_PD(_PGRAPH_PRI_SSYNC, _FSM_STATES);
    
#ifdef LW_PGRAPH_PRI_FE_CTXSW_USER_STATE0
    PRINT_REG_PD(_PGRAPH_PRI_FE, _CTXSW_USER_STATE0);
#endif

#ifdef LW_PGRAPH_CTXSW_STATUS
    PRINT_REG_PD(_PGRAPH_CTXSW, _STATUS);
#endif

    // LW_PGRAPH_PRI_FECS_LWRRENT/NEW_CTX
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _LWRRENT_CTX);
    PRINT_REG_PD(_PGRAPH_PRI_FECS, _NEW_CTX);
    
    PRINT_REG_PD(_PGRAPH_PRI_FE, _GO_IDLE_TIMEOUT);
    
    // CROP/ZROP status for each FBP
    dprintf("====================\n");
    dprintf("Per FBP CROP/ZROP detailed status\n");
    dprintf("====================\n");

    numActiveFbp = pGr[indexGpu].grGetNumActiveFbp();
    for (fbpId = 0 ; fbpId < numActiveFbp ; fbpId++)
    {
        PRINT_BE_REG_PD(CROP_STATUS*, fbpId);
        PRINT_BE_REG_PD(ZROP_STATUS*, fbpId);
    }

    //dump xbar related regs
    pGr[indexGpu].grDumpXbarStatus( grIdx );

    //niso hub status
    pGr[indexGpu].grDumpNisoHubStatus();

    //2x Tex Signature Dump
    pGr[indexGpu].grGetTexHangSignatureInfoLww( grIdx );
}

// LW_PFIFO_ENGINE_STATUS() has a different stride starting on Kepler
void grPrintEngineGraphicsStatus_GA100(void)
{
    /* Need to add call to GPU_REG_RD32(pGpu, priBase +
     * LW_RUNLIST_ENGINE_STATUS0(rlEngId)); as mentioned
     * in fifoGetEngineStatus_GA100 in resman which
     * correspond to info in LW_PFIFO_ENGINE_STATUS(0).
     * While other dependent HAL additions are also needed so
     * difered for next patch.
     */
    dprintf("grPrintEngineGraphicsStatus to be implemented for Ampere and later.\n");
}

