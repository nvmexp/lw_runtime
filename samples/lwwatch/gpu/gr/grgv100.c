/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgv100.c
//
//*****************************************************

//
// includes
//
#include <time.h>

#include "chip.h"
#include "utils/lwassert.h"
#include "volta/gv100/hwproject.h"
#include "volta/gv100/dev_graphics_nobundle.h"
#include "volta/gv100/dev_gpc_no_tpc.h"
#include "volta/gv100/dev_tpc.h"
#include "inst.h"
#include "print.h"
#include "gpuanalyze.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

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
     DFD_VAL(AUTO_INC_RD_END_ROW  , 0x1f       )      | \
     DFD_DEF(AUTO_INC_RD_ITER0    , LANE       )      | \
     DFD_DEF(AUTO_INC_RD_ITER1    , SUBP       )      | \
     DFD_DEF(AUTO_INC_RD_ITER2    , ROW        ))

#define SM_DFD_CONTROL_WARP_PC_READ_TPC_EXP             \
     DFD_VAL(AUTO_INC_RD_END_ROW  , 0x1f       )


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

LwU32 grGetNumSmPerTpc_GV100(void)
{
    return LW_SCAL_LITTER_NUM_SM_PER_TPC;
}

void grDumpTPCSMExceptionState_GV100(LwU32 gpcIdx, LwU32 tpcIdx)
{
    LwU32 hwwWarpEsr, hwwWarpRptMask, hwwGlbEsr, hwwGlbRptMask;
    LwU32 eccStatus;
    LwU32 smIdx, smOffset, numSmPerTpc;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcIdx,
            GR_UNIT_TYPE_TPC, tpcIdx),
        return);

    numSmPerTpc = pGr[indexGpu].grGetNumSmPerTpc();

    dprintf(" _TPC0_TPCCS_TPC_EXCEPTION_SM_PENDING\n");

    for (smIdx = 0; smIdx < numSmPerTpc; smIdx++)
    {
        // each SM within a TPC is at this address stride from SM0
        smOffset = smIdx * LW_SM_PRI_STRIDE;

        hwwWarpEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM0_HWW_WARP_ESR + smOffset);
        hwwWarpRptMask = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM0_HWW_WARP_ESR_REPORT_MASK + smOffset);

        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR :0x%x", gpcIdx, tpcIdx, smIdx, hwwWarpEsr);
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_REPORT_MASK :0x%x\n", gpcIdx, tpcIdx, smIdx, hwwWarpRptMask);

        dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR 0x%x\n", gpcIdx, tpcIdx, smIdx, hwwWarpEsr);
        dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_REPORT_MASK 0x%x\n", gpcIdx, tpcIdx, smIdx, hwwWarpRptMask);
        dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_WARP_ID = %d\n", gpcIdx, tpcIdx, smIdx,
                    DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _WARP_ID, hwwWarpEsr));

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_NONE)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_NONE\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_API_STACK_ERROR)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_API_STACK_ERROR\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_MISALIGNED_PC)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_MISALIGNED_PC\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_PC_OVERFLOW)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_PC_OVERFLOW\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_MISALIGNED_REG)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_MISALIGNED_REG)\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_ENCODING)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_ENCODING\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_OOR_REG)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_OOR_REG\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_OOR_ADDR)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_OOR_ADDR\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_MISALIGNED_ADDR)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_MISALIGNED_ADDR\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_ILWALID_ADDR_SPACE)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_ILWALID_ADDR_SPACE\n", gpcIdx, tpcIdx, smIdx);
        }

        if (DRF_VAL(_PTPC, _PRI_SM0_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM0_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR_LDC)
        {
            dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR_LDC\n", gpcIdx, tpcIdx, smIdx);
        }

        // Handle GLOBAL Error
        hwwGlbEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM0_HWW_GLOBAL_ESR + smOffset);
        hwwGlbRptMask = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM0_HWW_GLOBAL_ESR_REPORT_MASK + smOffset);

        dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_GLOBAL_ESR 0x%x\n", gpcIdx, tpcIdx, smIdx, hwwGlbEsr);
        dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_GLOBAL_ESR_REPORT_MASK 0x%x\n", gpcIdx, tpcIdx, smIdx, hwwGlbRptMask);


        if (hwwGlbEsr & DRF_DEF(_PTPC, _PRI_SM0_HWW_GLOBAL_ESR, _MULTIPLE_WARP_ERRORS, _PENDING))
        {
            dprintf("Graphics SM HWW Global Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM%d_HWW_GLOBAL_ESR_PHYSICAL_MULTIPLE_WARP_ERRORS\n", gpcIdx, tpcIdx, smIdx);
        }
    }

    eccStatus = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_LRF_ECC_STATUS);

    if (eccStatus)
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM_LRF_ECC_STATUS :0x%x -- TODO: Lwrrently not supported on GVXXX\n", gpcIdx, tpcIdx, eccStatus);
    }
    // See _GF100 when adding ECC code here
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
    dprintf ("LW_PGRAPH_PRI_GPC%u_TPC%u_SM1_DBGR_WARP_VALID_MASK_1 @(0x%.8x) = 0x%.8x\n", gpcId, tpcId,SM_REG_ADDR (DBGR_WARP_VALID_MASK_1, gpcId, tpcId, 1), warpValidMaskPtr [3]);
    warpValid |= warpValidMaskPtr [3];

    if (warpValid)
    {
        if (repeatPerWarp)
        {
            char *inactiveWarp = " (i)";
            LwBool tpcMoving = LW_FALSE;

            dprintf("Warp PCs for GPC %d, TPC %d, (i) == inactive warp\n", gpcId, tpcId );
            dprintf("%12s %12s %7s %4s %18s %5s %10s %13s %6s\n", "SM Warp ID", "Subpartition", "Warp ID", "VEID", "Warp PC", "Valid", "Run State", "Barrier State", "Repeat" );

            for ( repeat = 0; repeat < repeatPerWarp; repeat += 1)
            {
                // Turn on the DFD machine and set up the warp data read
                // pattern.  Only print data for active warps, but do the reads
                // for every warp so the DFD auto-increment happens.
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_0);
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), SM_DFD_CONTROL_WARP_PC_READ_TPC_WARPS_1);

                for ( warpId = 0 ; warpId < (warpCount / LW_PTPC_PRI_SM_ARCH_SUBPARTITION) ; warpId ++ )
                {
                    for ( subPartitionId = 0 ; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION ; subPartitionId ++ )
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

                        if (lastPC == warpPC && lastRunState == runState && lastBarrierState == barrierState)
                        {
                            pcRepeatCount += 1;
                        }
                        else {
                            if (pcRepeatCount) {
                                dprintf(" x %4d", pcRepeatCount + 1);
                            }

                            pcRepeatCount = 0;

                            if (repeat)
                            {
                                dprintf("\n%33s", " ");
                            }

                            dprintf( " %4d 0x%.16llx %5s 0x%.8x    0x%.8x", veId, warpPC, warpValidStr, runState, barrierState);
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

void grDumpWarpPc_GV100(LwU32 gpcId, LwU32 tpcId, LwU32 repeat, LwBool bAnnotate)
{
    _grDumpTpcWarpPcs(gpcId, tpcId, repeat, bAnnotate);
}

// This method is called for all Volta+ chips.
void grDumpTpcInfo_GV100(LwU32 gpcId, LwU32 tpcId)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 data32, addr32;
    LwU32 smId;

    GR_IO_APERTURE *pTpcAperture = NULL;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcId, GR_UNIT_TYPE_TPC, tpcId),
        return);

    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("%s GPC/TPC %d/%d\n", GpuArchitecture(), gpcId, tpcId);
    dprintf("====================\n");

    PRINT_TPC_REG_PD(TPCCS_TPC_ACTIVITY0, gpcId, tpcId);

    PRINT_TPC_REG_PD(PE_STATUS, gpcId, tpcId);

    DUMP_TPC_APERTURE_REG(pTpcAperture, PE_L2_EVICT_POLICY);
    DUMP_TPC_APERTURE_REG(pTpcAperture, PE_HWW_ESR);

    // These registers hold 3 bit values in ACTIVITY format but those
    // encodings aren't enumerated in the manuals, so priv_dump just
    // prints the numeric values.
    PRINT_TPC_REG_PD(MPC_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_VTG_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_PIX_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_WLU_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_COMP_STATUS, gpcId, tpcId);

    pGr[indexGpu].grDumpWarpPc( gpcId, tpcId, 1, LW_FALSE );


    PRINT_TPC_REG_PD ( SM_BLK_ACTIVITY_PRIV_LEVEL_MASK, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG2, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_TEXIO_CONTROL, gpcId, tpcId );

    PRINT_TPC_REG_PD ( SM_PM_SAMP_CTRL, gpcId, tpcId );

    DUMP_TPC_APERTURE_REG(pTpcAperture, SM_CONFIG);
    PRINT_TPC_REG_PD(SM_ARCH, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_CACHE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_PRIVATE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_DISP_CTRL, gpcId, tpcId);
    DUMP_TPC_APERTURE_REG(pTpcAperture, SM_MACHINE_ID0);
    // See _GF100 when adding ECC code here
    PRINT_TPC_REG_PD(SM_DEBUG_SFE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_QUAD_BA_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_THROTTLE_CTL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_VST_CTL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_VST_DATA, gpcId, tpcId);

    // Implementation of bug #1374564
    // TEX PRIs route to PIPE 0 by default
    addr32 = REG_GET_ADDR( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
    data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
        LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
    REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );

    for (smId = 0; smId < 2; smId++)
    {
        if ( smId == 1 )
        {      
            // Route to PIPE 1
            data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
        }
      
        dprintf("==========================\n");
        dprintf("%s GPC/TPC/SM %d/%d/%d\n", GpuArchitecture(), gpcId, tpcId, smId);
        dprintf("==========================\n");

        // CYA bits in the SM LW_PGRAPH_PRI_GPC#0_TPC#_SM_HALFCTL_CTRL
        PRINT_TPC_REG_PD(SM_HALFCTL_CTRL, gpcId, tpcId);

        PRINT_SM_REG_PD(INFO_SCTL_SUBUNIT_STATUS, gpcId, tpcId, smId);
        PRINT_SM_REG_PD(INFO_L1C_SUBUNIT_STATUS,  gpcId, tpcId, smId);

        PRINT_TPC_REG_PD(TEX_M_ROUTING, gpcId, tpcId);;
        PRINT_TPC_REG_PD(TEX_M_TEX_SUBUNITS_STATUS, gpcId, tpcId);

        // Subunit and subunit_half status
        PRINT_TPC_REG_PD( SM_STATUS, gpcId, tpcId );
        PRINT_TPC_REG_PD( SM_SFE_STATUS, gpcId, tpcId );

        DUMP_SM_REG(HWW_ESR_ADDR_0, gpcId, tpcId, smId);
        DUMP_SM_REG(HWW_ESR_ADDR_1, gpcId, tpcId, smId);

        PRINT_SM_REG_PD(HWW_WARP_ESR_REPORT_MASK, gpcId, tpcId, smId);
        PRINT_SM_REG_PD(HWW_WARP_ESR, gpcId, tpcId, smId);
        PRINT_SM_REG_PD(HWW_GLOBAL_ESR_REPORT_MASK, gpcId, tpcId, smId);
        PRINT_SM_REG_PD(HWW_GLOBAL_ESR, gpcId, tpcId, smId);


    }
    dprintf ("\n");
}

LwU32 grGetNumberPesPerGpc_GV100(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetMaxTpcPerGpc_GV100()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

/*----------------------------------------------------------------------------
 * Get count of total number of mailbox register for a given chip from dev_graphics_nobundle.h
 * and print all the mailbox registers
 *----------------------------------------------------------------------------
 */
void grDumpFecsMailboxRegisters_GV100(void)
{
    LwU32 regMbCnt;
    char    buffer[GR_REG_NAME_BUFFER_LEN];
    dprintf("FECS MAILBOX registers:\n");
    for(regMbCnt=0; regMbCnt< LW_PGRAPH_PRI_FECS_CTXSW_MAILBOX__SIZE_1; regMbCnt++)
    {
        sprintf( buffer, "LW_PGRAPH_PRI_FECS_CTXSW_MAILBOX(%d)",regMbCnt);
        priv_dump(buffer);
    }
}

//HSHUB
void grDumpHshubIdleRegisters_GV100(void)
{
    //Always dump following registers as part of detailed report
    dprintf("\n==========\nPFB HSHUB IG/EG/RR IDLE0/1 registers\n==========\n");
    PRINT_REG_PD(_PFB, _HSHUB_IG_IDLE*);
    PRINT_REG_PD(_PFB, _HSHUB_EG_IDLE*);
    PRINT_REG_PD(_PFB, _HSHUB_RR_IDLE*);
}

/*----------------------------------------------------------------------------
 * static void
 * grDumpConsolidatedReport_GV100()
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
void grDumpConsolidatedReport_GV100( LwU32 grIdx )
{
    LwU32 grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act3, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, GM200);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GV100 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GV100 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GV100 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GV100 );

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
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
        }
    }

    // Per FBP
    DUMP_REG(ACTIVITY3);
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
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
            PRINT_BE_REG_Z( i, BECS_BE_ACTIVITY0, GV100 );
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

void grDumpConsolidatedReportGpc_GV100( LwU32 numActiveGpc, LwU32 grIdx )
{
    LwU32 tpcId, numActiveTpc, gpcId, grStatus, val;
    LwU32 data32;
    char buffer[GR_REG_NAME_BUFFER_LEN];
    GR_IO_APERTURE *pGpcAperture;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;

    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcId),
            return);

        dprintf("====================\n");
        dprintf("GPC %d\n", gpcId);
        dprintf("====================\n");

        numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY0, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY1, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY2, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY3, GV100 );

        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_1, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_GPC_0, GV100 );

        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {
            LW_ASSERT_OK_OR_ELSE(status,
                GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcId,
                    GR_UNIT_TYPE_TPC, tpcId),
                return);

            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, TPCCS_TPC_ACTIVITY0, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, PE_STATUS, GM200 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_VTG_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_PIX_STATUS, GV100 );

            data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/0\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, GV100 );

            data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/1\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, GV100 );
        }
    }
}

void grDumpGccExceptionState_GV100(LwU32 gpcCounter)
{
    //LwU32 hwwEsr;
    //LwU32 badHdrIdx, badSmpIdx;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("====================\n");
    dprintf("GPCCS Exception State detailed status\n");
    dprintf("====================\n");

    GPU_REG_RD32(GPC_REG_ADDR(GCC_HWW_ESR, gpcCounter));
    //badHdrIdx = GPU_REG_RD32(GPC_REG_ADDR(GCC_BAD_TEX_HDR_INDEX, gpcCounter));
    //badSmpIdx = GPU_REG_RD32(GPC_REG_ADDR(GCC_BAD_TEX_SMP_INDEX, gpcCounter));

    DUMP_GPC_REG(GCC_HWW_ESR, gpcCounter );

}

void grDumpSetupExceptionState_GV100(LwU32 gpcCounter)
{
    LwU32 hwwEsr;
    char buffer[GR_REG_NAME_BUFFER_LEN];


    dprintf("====================\n");
    dprintf("Setup Exception State detailed status\n");
    dprintf("====================\n");

    hwwEsr = GPU_REG_RD32(GPC_REG_ADDR(SETUP_HWW_ESR, gpcCounter));
    DUMP_GPC_REG(SETUP_HWW_ESR, gpcCounter);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _LDCONST_OOB, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_LDCONST_OOB\n", gpcCounter);
        dprintf("VRM: Graphics SETUP Exception Type: LDCONST_OOB\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_GPC0_SETUP_HWW_ESR, _POOLPAGE_ILWALID, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_SETUP_HWW_ESR_POOLPAGE_ILWALID\n", gpcCounter);
        dprintf("Graphics SETUP Exception Type: POOLPAGE_ILWALID\n");
    }
}

/*!
 * @brief Provides the caller with information about a particular type of GR Aperture.
 *
 * Output parameters may be NULL if that particular aperture information is not
 * required.
 *
 * @param[in]  pGpu        OBJGPU pointer
 * @param[in]  pGr         OBJGR pointer
 * @param[in]  type        type of the Aperture, GR_UNIT_TYPE* macros defined in grunits.h
 * @param[out] pUnitBase   Base address for the first unit Aperture of its kind
 * @param[out] pUnitStride Stride length for the scalable unit
 * @param[out] pUnitBCIdx  Signed index for a broadcast Aperture, relative to Base*
 *
 * @return LW_STATUS LW_OK upon success
 *                   LW_ERR_NOT_SUPPORTED for an unknown aperture type for this Arch.
 */
LW_STATUS
grGetUnitApertureInformation_GV100
(
    GR_UNIT_TYPE type,
    LwU32       *pUnitBase,
    LwU32       *pUnitStride,
    LwS32       *pUnitBCIdx
)
{
    LwU32 unused;
    pUnitStride = (pUnitStride != NULL) ? pUnitStride : &unused;

    if (type != GR_UNIT_TYPE_ROP)
    {
        pUnitBCIdx = (pUnitBCIdx != NULL) ? pUnitBCIdx : (LwS32 *)&unused;
        pUnitBase  = (pUnitBase  != NULL) ? pUnitBase  : &unused;
    }

    switch(type)
    {
        case GR_UNIT_TYPE_GR:
            *pUnitBase = DRF_BASE(LW_PGRAPH);
            *pUnitStride = DRF_SIZE(LW_PGRAPH);
            *pUnitBCIdx = 0;
            break;

        case GR_UNIT_TYPE_GPC:
            *pUnitBase = LW_GPC_PRI_BASE - DRF_BASE(LW_PGRAPH); // Find GPC in GR Base
            *pUnitStride = LW_GPC_PRI_STRIDE;
            *pUnitBCIdx = (LW_GPC_PRI_SHARED_BASE - LW_GPC_PRI_BASE) / LW_GPC_PRI_STRIDE;
            break;

        // ETPC,TPC units have identical layouts within their parent's PRI space.
        case GR_UNIT_TYPE_ETPC:
        case GR_UNIT_TYPE_TPC:
            *pUnitBase = LW_TPC_IN_GPC_BASE;
            *pUnitStride = LW_TPC_IN_GPC_STRIDE;
            *pUnitBCIdx = (LW_TPC_IN_GPC_SHARED_BASE - LW_TPC_IN_GPC_BASE) / LW_TPC_IN_GPC_STRIDE;
            break;

        case GR_UNIT_TYPE_ROP:
            //
            // For pre-GA10X chips, ROP is not in GPC so we don't get the base here
            // We support stride here to avoid forking more HALs
            //
            if ((pUnitBCIdx != NULL) || (pUnitBase != NULL))
            {
                return LW_ERR_NOT_SUPPORTED;
            }

            *pUnitStride = LW_ROP_PRI_STRIDE;
            break;

        case GR_UNIT_TYPE_PPC:
            *pUnitBase = LW_PPC_IN_GPC_BASE;
            *pUnitStride = LW_PPC_IN_GPC_STRIDE;
            *pUnitBCIdx = (LW_PPC_IN_GPC_SHARED_BASE - LW_PPC_IN_GPC_BASE) / LW_PPC_IN_GPC_STRIDE;
            break;

        case GR_UNIT_TYPE_EGPC:
            *pUnitBase = LW_EGPC_IN_GR_BASE;
            *pUnitStride = LW_EGPC_PRI_STRIDE;
            *pUnitBCIdx = LW_EGPC_PRI_SHARED_INDEX;
            break;

        default:
            return LW_ERR_NOT_SUPPORTED;
    }

    return LW_OK;
}

/*!
 * @brief A constructor for EGPC Apertures.
 *
 * @param[in] pGrAperture pointer to GR IO_APERTURE instance.
 * @param[in] bIsExtended if true EGPC will be constructed.
 *
 * @return LW_OK upon success.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures.
 *         LW_ERR_NOT_SUPPORTED when a subunit count is determined to be 0.
 *         LW_ERR* for other failures.
 */
LW_STATUS
grConstructEgpcAperturesStruct_GV100
(
    GR_IO_APERTURE *pGrAperture
)
{
    LW_STATUS status = LW_OK;

    // Create apertures for GPC registers.
    LW_ASSERT_OK_OR_GOTO(status,
        pGr[indexGpu].grConstructGpcApertures(pGrAperture, LW_TRUE),
        done);

    return LW_OK;

done:
    if (status != LW_OK)
    {
        pGr[indexGpu].grDestroyIoApertures(pGrAperture);
    }

    return status;
}
