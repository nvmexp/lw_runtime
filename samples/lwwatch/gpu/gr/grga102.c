/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grga102.c
//
//*****************************************************

//
// includes
//
#include "chip.h"
#include "inst.h"
#include "print.h"
#include "utils/lwassert.h"
#include "ampere/ga102/dev_graphics_nobundle.h"
#include "ampere/ga102/dev_rop.h"
#include "ampere/ga102/dev_gpc_no_tpc.h"
#include "ampere/ga102/dev_tpc.h"
#include "ampere/ga102/hwproject.h"
#include "ampere/ga102/dev_fuse.h"
#include "ampere/ga102/dev_top.h"
#include "gr.h"
#include "g_gr_private.h"  // (rmconfig) implementation prototypes

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

typedef struct
{
  LwU32 pc_lower;
  LwU32 pc_upper;
  LwU32 run_state;
  LwU32 barrier_state;
} SMDfdWarpData;

/*!
 * @brief The constructor for ROP Apertures.
 *
 * @param[in] pGrAperture pointer to GR instance.
 * @param[in] pAperture pointer to the parent GPC aperture.
 *
 * Compute only has some special cases that are handled w/o returning
 * an error.
 *
 * when setting up broadcase
 *
 * @return LW_OK upon success.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures.
 *         LW_ERR* for other failures.
 */
LW_STATUS
grConstructRopAperturesStruct_GA102
(
    GR_IO_APERTURE *pGrAperture,
    GR_IO_APERTURE *pGpcAperture
)
{
    LwU32 ropInGpcCount;
    LwU32 grIdx = pGrAperture->unitIndex; // Not untilized by grGetNumRopForGpc()
    LwU32 gpcIdx = pGpcAperture->unitIndex;

    if (pGpcAperture->bIsBroadcast)
    {
        LwU32 i;
        LwU32 minRopCount = pGr[indexGpu].grGetNumRopForGpc(0, grIdx);

        for (i = 1; i < pGrAperture->unitCounts[GR_UNIT_TYPE_GPC]; i++)
        {
            LwU32 ropCount = pGr[indexGpu].grGetNumRopForGpc(i, grIdx);

            // Skip compute only GPCs
            if (ropCount != 0)
                minRopCount = LW_MIN(minRopCount, ropCount);
        }
        ropInGpcCount = minRopCount;
    }
    else
    {
        ropInGpcCount = pGr[indexGpu].grGetNumRopForGpc(gpcIdx, grIdx);
    }

    // Construct ROP Apertures.
    LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructSubApertures(pGpcAperture,
        GR_UNIT_TYPE_ROP, ropInGpcCount));

    return LW_OK;
}

LwU32
grGetActiveRopsForGpc_GA102(LwU32 gpcIdx)
{
    LwU32 regVal;
    regVal = GPU_REG_IDX_RD_DRF(_FUSE, _STATUS_OPT_ROP_GPC, gpcIdx, _DATA);
    return ~regVal & (LWBIT32(LW_SCAL_LITTER_NUM_ROP_PER_GPC) - 1);
}

LwU32 grGetNumRopForGpc_GA102(LwU32 gpcIdx, LwU32 grIdx)
{
    LwU32 reg;
    const LwU32 num_gpcs = pGr[indexGpu].grGetNumGpcs( grIdx );
    GR_IO_APERTURE *pGpcAperture = NULL;
    LW_STATUS status;

    if (gpcIdx >= num_gpcs)
    {
        dprintf("**ERROR: Illegal GPC num: %d (%s)\n", gpcIdx, __FUNCTION__);
        return 0;
    }

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcIdx),
        return 0);

    reg = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_GPCCS_FS_GPC);
    return DRF_VAL(_PGPC_PRI, _GPCCS_FS_GPC, _NUM_AVAILABLE_ROPS, reg);
}

/*!
 * @brief Provides the caller with information about a particular type of GR Aperture
 *
 * Output parameters may be NULL if that particular aperture information is not
 * required.
 *
 * @param[in]  type        type of the Aperture, GR_UNIT_TYPE* macros defined in grunits.h
 * @param[out] pUnitBase   Base address for the first unit Aperture of its kind
 * @param[out] pUnitStride Stride length for the scalable unit
 * @param[out] pUnitBCIdx  Signed index for a broadcast Aperture, relative to Base
 *
 * @return LW_STATUS LW_OK upon success
 *                   LW_ERR_ILWALID_ARGUMENT for an unknown aperture type for this Arch.
 */
LW_STATUS
grGetUnitApertureInformation_GA102
(
    GR_UNIT_TYPE type,
    LwU32       *pUnitBase,
    LwU32       *pUnitStride,
    LwS32       *pUnitBCIdx
)
{
    LwU32 unused;
    pUnitBase   = (pUnitBase   != NULL) ? pUnitBase   : &unused;
    pUnitStride = (pUnitStride != NULL) ? pUnitStride : &unused;
    pUnitBCIdx  = (pUnitBCIdx  != NULL) ? pUnitBCIdx  : (LwS32 *)&unused;

    switch (type)
    {
        case GR_UNIT_TYPE_GR:
            *pUnitBase = DRF_BASE(LW_PGRAPH);
            *pUnitStride = DRF_SIZE(LW_PGRAPH);
            *pUnitBCIdx = 0;
            break;

        case GR_UNIT_TYPE_GPC:
            *pUnitBase = LW_GPC_IN_GR_BASE;
            *pUnitStride = LW_GPC_PRI_STRIDE;
            *pUnitBCIdx = LW_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_EGPC:
            *pUnitBase = LW_EGPC_IN_GR_BASE;
            *pUnitStride = LW_EGPC_PRI_STRIDE;
            *pUnitBCIdx = LW_EGPC_PRI_SHARED_INDEX;
            break;

        // ETPC,TPC units have identical layouts within their parent's PRI space.
        case GR_UNIT_TYPE_ETPC:
        case GR_UNIT_TYPE_TPC:
            *pUnitBase = LW_TPC_IN_GPC_BASE;
            *pUnitStride = LW_TPC_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_TPC_IN_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_ROP:
            *pUnitBase = LW_ROP_IN_GPC_BASE;
            *pUnitStride = LW_ROP_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_ROP_IN_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_PPC:
            *pUnitBase = LW_PPC_IN_GPC_BASE;
            *pUnitStride = LW_PPC_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_PPC_IN_GPC_PRI_SHARED_INDEX;
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }

    return LW_OK;
}

void grDumpPgraphExceptionsState_GA102(LwU32 grIdx)
{
    LwU32   regExc = GPU_REG_RD32(LW_PGRAPH_EXCEPTION);
    LwU32   regEn = GPU_REG_RD32(LW_PGRAPH_EXCEPTION_EN);
    LwU32   data32 = 0;

    grDumpPgraphExceptionsState_GK104(grIdx);

    //MME_FE1
    if (DRF_VAL(_PGRAPH, _EXCEPTION, _MME_FE1, regExc) == LW_PGRAPH_EXCEPTION_MME_FE1_PENDING)
    {
        dprintf("        _EXCEPTION_MME_FE1_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_MME_FE1_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_FE1_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _MISSING_MACRO_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_MISSING_MACRO_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_MISSING_MACRO_DATA_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _EXTRA_MACRO_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_EXTRA_MACRO_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_EXTRA_MACRO_DATA_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _ILLEGAL_OPCODE,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_ILLEGAL_OPCODE_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_ILLEGAL_OPCODE_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _BRANCH_IN_DELAY_SLOT, _PENDING))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_BRANCH_IN_DELAY_SLOT_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_BRANCH_IN_DELAY_SLOT_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _MAX_INSTR_LIMIT, _PENDING))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_MAX_INSTR_LIMIT_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_MAX_INSTR_LIMIT_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _STOP_ON_TRAP,_DISABLED))
        {
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_STOP_ON_TRAP_DISABLED\n");
            dprintf("Now checking _PRI_MME_FE1_ESR_INFO/INFO2\n");

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_FE1_HWW_ESR_INFO);
            if (!(data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _INFO_PC_VALID, _INIT)))
            {
                dprintf("            _PRI_MME_FE1_HWW_ESR_INFO_PC: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _MME_FE1_HWW_ESR_INFO, _PC, data32));
            }

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_FE1_HWW_ESR_INFO2);
            dprintf("            _PRI_MME_FE1_HWW_ESR_INFO2_IR: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _MME_FE1_HWW_ESR_INFO2, _IR, data32));
        }
        else
        {
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_STOP_ON_TRAP_ENABLED\n");
        }

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _EN,_ENABLE))
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_EN_ENABLE\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_FE1_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_MME_FE1_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_MME_FE1_HWW_ESR_RESET_ACTIVE\n");
        }
    }//end MME_FE1 EXCEPTION
}

void grPrintMmeFe1Disabled_GA102(LwU32 regEn)
{
    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN, _MME_FE1, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_MME_FE1_DISABLED)
        dprintf("         _EXCEPTION_EN_MME_FE1_DISABLED\n");

}

void grDumpPropExceptionState_GA102(LwU32 gpcIdx)
{
    LwU32 hwwEsr, esrCoord, esrFormat, esrState;
    GR_IO_APERTURE *pGpcAperture;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcIdx),
        return);
    hwwEsr = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_PROP_HWW_ESR);
    esrCoord = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_PROP_HWW_ESR_COORD);
    esrFormat = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_PROP_HWW_ESR_FORMAT);
    esrState = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_PROP_HWW_ESR_STATE);

    grDumpPropExceptionState_GK104(gpcIdx);

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _TIR_ZBUF_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_TIR_ZBUF_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP TIR ZBUF Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    COLOR FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _CT, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }
}

// Gets active FBP config.
LwBool grGetActiveFbpaConfig_GA102(LwU32 *activeFbpaConfig, LwU32 *maxNumberOfFbpas)
{
    LwU32 activeFbpMask;

    activeFbpMask = GPU_REG_RD32(LW_FUSE_STATUS_OPT_FBP);
    if ((*activeFbpaConfig & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_FUSE_STATUS_OPT_FBP register read gave 0x%x value.\n", *activeFbpaConfig);
        return LW_FALSE;
    }
    *maxNumberOfFbpas = GPU_REG_RD32(LW_PTOP_SCAL_NUM_FBPAS);
    if ((*maxNumberOfFbpas & 0xFFFF0000) == 0xBADF0000)
    {
        dprintf ("FATAL ERROR! LW_PTOP_SCAL_NUM_FBPAS register read gave 0x%x value.\n", *maxNumberOfFbpas);
        return LW_FALSE;
    }

    activeFbpMask = ~activeFbpMask & BITMASK(*maxNumberOfFbpas);
    *activeFbpaConfig = activeFbpMask;

    return LW_TRUE;
}

//compared to function in grga100.c, no print of LW_PGRAPH_STATUS2 starting from GA102, no FBP CROP/ZROP status
void grDumpDetailedStatus_GA102( BOOL bFullPrint, LwU32 grIdx )
{
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

    //dump xbar related regs
    pGr[indexGpu].grDumpXbarStatus( grIdx );

    //niso hub status
    pGr[indexGpu].grDumpNisoHubStatus();

    //2x Tex Signature Dump
    pGr[indexGpu].grGetTexHangSignatureInfoLww( grIdx );
}

//compared to funtion in grgp100.c, add ROP status as per RIG change
void grDumpGpcInfo_GA102 ( LwU32 gpcId, LwU32 grIdx )
{
    LwU32 tpcId, ropId, numActiveTpc, numActiveRop;
    char buffer[GR_REG_NAME_BUFFER_LEN] = { 0 };
    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("GPC %d\n", gpcId);
    dprintf("====================\n");

    numActiveRop = pGr[indexGpu].grGetNumRopForGpc( gpcId, grIdx );
    for ( ropId = 0 ; ropId < numActiveRop ; ropId++ )
    {
        PRINT_ROP_REG_PD(ZROP_STATUS*, gpcId, ropId);
        PRINT_ROP_REG_PD(ZROP_DEBUG_COUNT_REQUESTS, gpcId, ropId);
        PRINT_ROP_REG_PD(ZROP_DEBUG_COUNT_RESPONSES, gpcId, ropId);
        PRINT_ROP_REG_PD(RRH_STATUS, gpcId, ropId);
        PRINT_ROP_REG_PD(CROP_STATUS*, gpcId, ropId);
        PRINT_ROP_REG_PD(CROP_DEBUG_COUNT_REQUESTS, gpcId, ropId);
        PRINT_ROP_REG_PD(CROP_DEBUG_COUNT_RESPONSES, gpcId, ropId);
    }

    // '*' in the regex matches all ACTIVITY# registers for this GPC
    PRINT_GPC_REG_PD(GPCCS_GPC_ACTIVITY*, gpcId);

    DUMP_GPC_REG(PROP_HWW_ESR, gpcId);
    DUMP_GPC_REG(PROP_HWW_ESR_COORD, gpcId);
    DUMP_GPC_REG(PROP_HWW_ESR_FORMAT, gpcId);
    DUMP_GPC_REG(PROP_PS_ILWOCATIONS_HI, gpcId);

    DUMP_GPC_REG(PROP_ZPASS_CNT_LO, gpcId);
    DUMP_GPC_REG(PROP_ZPASS_CNT_HI, gpcId);

    DUMP_GPC_REG(PROP_EZ_ZPASS_CNT_LO, gpcId);
    DUMP_GPC_REG(PROP_EZ_ZPASS_CNT_HI, gpcId);
    // Use priv_dump with regex to print all STATE_* regs.  GM20x added _6 and _7
    PRINT_GPC_REG_PD(PROP_STATE_., gpcId);

    DUMP_GPC_REG(PROP_PM_CTRL, gpcId);
    DUMP_GPC_REG(PROP_CG, gpcId);
    DUMP_GPC_REG(PROP_CG1, gpcId);

    DUMP_GPC_REG(FRSTR_DEBUG, gpcId);
    DUMP_GPC_REG(FRSTR_PM_CTRL, gpcId);
    DUMP_GPC_REG(FRSTR_CG1, gpcId);

    DUMP_GPC_REG(WIDCLIP_PM_CTRL, gpcId);
    DUMP_GPC_REG(WIDCLIP_DEBUG, gpcId);
    DUMP_GPC_REG(WIDCLIP_CG, gpcId);
    DUMP_GPC_REG(WIDCLIP_CG1, gpcId);

    DUMP_GPC_REG(SETUP_CG, gpcId);
    DUMP_GPC_REG(SETUP_CG1, gpcId);
    DUMP_GPC_REG(SETUP_DEBUG_LIMIT0, gpcId);
    DUMP_GPC_REG(SETUP_DEBUG_LIMIT1, gpcId);
    DUMP_GPC_REG(SETUP_PM_CTRL, gpcId);
    DUMP_GPC_REG(SETUP_HWW_ESR, gpcId);

    DUMP_GPC_REG(CRSTR_DEBUG, gpcId);
    DUMP_GPC_REG(CRSTR_MAP_TABLE_CONFIG, gpcId);
    DUMP_GPC_REG(CRSTR_CG, gpcId);

    DUMP_GPC_REG(ZLWLL_HWW_ESR, gpcId);
    DUMP_GPC_REG(ZLWLL_HWW_ESR_INFO_2, gpcId);
    DUMP_GPC_REG(ZLWLL_FS, gpcId);
    DUMP_GPC_REG(ZLWLL_CG, gpcId);
    DUMP_GPC_REG(ZLWLL_CG1, gpcId);

    PRINT_GPC_REG_PD(GPM_PD_STATUS, gpcId);
    PRINT_GPC_REG_PD(GPM_PD_STATUS2, gpcId);

    DUMP_GPC_REG(GPM_PD_PM, gpcId);
    DUMP_GPC_REG(GPM_PD_CG1, gpcId);
    DUMP_GPC_REG(GPM_SD_CONFIG, gpcId);

    PRINT_GPC_REG_PD(GPM_SD_STATUS, gpcId);

    numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc( gpcId, grIdx );
    for ( tpcId = 0 ; tpcId < numActiveTpc ; tpcId++ )
    {
        sprintf( buffer, "LW_PGRAPH_PRI_GPC%d_GPM_SD_STATUS_TPC(%d)", gpcId, tpcId );
        priv_dump( buffer );
    }

    DUMP_GPC_REG(GPM_SD_PM, gpcId);

    PRINT_GPC_REG_PD(GPM_SD_STATUS_COLL, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_DIST, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_IN, gpcId);
    PRINT_GPC_REG_PD(GPM_SD_STATUS_OUT, gpcId);

    DUMP_GPC_REG(GPM_SD_CG, gpcId);
    DUMP_GPC_REG(GPM_SD_CG1, gpcId);
    DUMP_GPC_REG(GPM_RPT_PM, gpcId);

    PRINT_GPC_REG_PD(GPM_RPT_ALPHA_WRITE_POINTERS, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_BETA_WRITE_POINTERS, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_0, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_1, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_2, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_SCOREBOARD_3, gpcId);
    PRINT_GPC_REG_PD(GPM_RPT_STATUS, gpcId);

    DUMP_GPC_REG(GPM_RPT_CG, gpcId);
    DUMP_GPC_REG(GPM_RPT_CG1, gpcId);

    DUMP_GPC_REG(WDXPS_PM, gpcId);
    DUMP_GPC_REG(WDXPS_CG, gpcId);
    DUMP_GPC_REG(WDXPS_CG1, gpcId);

    PRINT_GPC_REG_PD(SWDX_DEBUG, gpcId);

    DUMP_GPC_REG(SWDX_CG, gpcId);

    DUMP_GPC_REG(GCC_HWW_ESR, gpcId);
    DUMP_GPC_REG(GCC_CG1, gpcId);

    DUMP_GPC_REG(MMU_CTRL, gpcId);
    DUMP_GPC_REG(MMU_PM, gpcId);
    DUMP_GPC_REG(MMU_PM_ADDR_LOW, gpcId);
    DUMP_GPC_REG(MMU_PM_UNIT_MASK, gpcId);
    DUMP_GPC_REG(MMU_PM_REQ_MASK, gpcId);
    DUMP_GPC_REG(MMU_CG, gpcId);
    DUMP_GPC_REG(MMU_CG1, gpcId);
    DUMP_GPC_REG(MMU_PG, gpcId);
    DUMP_GPC_REG(MMU_PG1, gpcId);
    DUMP_GPC_REG(MMU_TLB_CTRL, gpcId);
    DUMP_GPC_REG(MMU_RSRVD, gpcId);
}

static void _grDumpTpcWarpPcs( LwU32 gpcId, LwU32 tpcId, LwU32 repeatPerWarp, LwBool bAnnotate )
{

    // Used by gr.h print macros
    char buffer[GR_REG_NAME_BUFFER_LEN];

    LwU32 warpGroupId, subPartitionId, smWarpId;
    LwU32 warpGroupBaseIdForSM1;
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

                // DFD Sequence (http://lwbugs/2740053/44)
                //
                // * GA100 (for reference) *
                //               -------------- warp traversal order ------------->
                //                 ...
                // Partition 0  | : 0 : 4  8  12 16 20 24 28 32 36 40 44 48 52 56 60
                // Partition 1  | : 1 : 5  9  13 17 21 25 29 33 37 41 45 49 53 57 61
                // Partition 2  | : 2 : 6  10 14 18 22 26 30 34 38 42 46 50 54 58 62
                // Partition 3  | : 3 : 7  11 15 19 23 27 31 35 39 43 47 51 55 59 63
                //                 ... <-- warp group
                // WarpGroup ID |   0   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
                //
                // On GA100, we capture the warp states incrementally one warp group at a time by repeatedly poking the
                // SM_DFD_DATA register. To collect the states of a single warp, we have to poke SM_DFD_DATA 4 times. So,
                // to cover warpGroup0, we would read the DFD register 4 warps * 4 reads/warp = 16 times. The subsequent
                // read of the DFD register would result in reading the state of warp4 from warpGroup1.
                //
                // * GA10x *
                //                                       |-> read and discard  <-|
                //                 ...                 /---\                   /---\.
                // Partition 0  | : 0 : 4  8  12 16 20 xx xx 24 28 32 36 40 44 xx xx
                // Partition 1  | : 1 : 5  9  13 17 21 xx xx 25 29 33 37 41 45 xx xx
                // Partition 2  | : 2 : 6  10 14 18 22 xx xx 26 30 34 38 42 46 xx xx
                // Partition 3  | : 3 : 7  11 15 19 23 xx xx 27 31 35 39 43 47 xx xx
                //                 ...                 \---/                   \---/
                // WarpGroup ID |   0   1  2  3  4  5  xx xx 6  7  8  9  10 11 xx xx
                //
                // DFD capture on GA10x is very similar to that on GA100, except that we have to discard the DFD reads
                // for the subsequent two warp groups after warpGroup5 and warpGroup11. In other words, when using
                // warpGroupID from GA100 figure (above) we have to ignore DFD reads for warpGroups {6, 7, 14, 15}.

                // Read DFD for SM0
                for ( warpGroupId = 0 ; warpGroupId < warpCount / (2 * LW_PTPC_PRI_SM_ARCH_SUBPARTITION) ; warpGroupId ++ )
                {
                    for (subPartitionId = 0; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION; subPartitionId++)
                    {
                        LwU32 warpDataIndex;
                        smWarpId = warpGroupId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                        warpDataIndex = smWarpId * repeatPerWarp + repeat;
                        warpDataBuffer[warpDataIndex].pc_lower      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].pc_upper      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].run_state     = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].barrier_state = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                    }

                    // Read and discard DFD reads for 2 warp groups after warpGroup5 and warpGroup11 (explanation above)
                    if (warpGroupId == 5 || warpGroupId == 11)
                    {
                        for (subPartitionId = 0; subPartitionId < 2 * LW_PTPC_PRI_SM_ARCH_SUBPARTITION * LW_PTPC_PRI_SM_DFD_LANES; subPartitionId++)
                        {
                            GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        }
                    }
                }

                // [Bug 2198150] Select pipe for reading from SM1
                GPU_REG_WR32(TPC_REG_ADDR(TEX_M_ROUTING, gpcId, tpcId), LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1);

                // Read DFD for SM1
                warpGroupBaseIdForSM1 = warpGroupId;
                for ( ; warpGroupId < warpCount / LW_PTPC_PRI_SM_ARCH_SUBPARTITION; warpGroupId ++ )
                {
                    for (subPartitionId = 0; subPartitionId < LW_PTPC_PRI_SM_ARCH_SUBPARTITION; subPartitionId++)
                    {
                        LwU32 warpDataIndex;
                        smWarpId = warpGroupId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                        warpDataIndex = smWarpId * repeatPerWarp + repeat;
                        warpDataBuffer[warpDataIndex].pc_lower      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].pc_upper      = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].run_state     = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        warpDataBuffer[warpDataIndex].barrier_state = GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                    }

                    // Read and discard DFD reads for 2 warp groups after warpGroup5 and warpGroup11 (explanation above)
                    if (warpGroupId == (warpGroupBaseIdForSM1 + 5) || warpGroupId == (warpGroupBaseIdForSM1 + 11))
                    {
                        for (subPartitionId = 0; subPartitionId < 2 * LW_PTPC_PRI_SM_ARCH_SUBPARTITION * LW_PTPC_PRI_SM_DFD_LANES; subPartitionId++)
                        {
                            GPU_REG_RD32(TPC_REG_ADDR(SM_DFD_DATA, gpcId, tpcId));
                        }
                    }
                }

                // Shut down DFD machine
                GPU_REG_WR32(TPC_REG_ADDR(SM_DFD_CONTROL, gpcId, tpcId), 0x0);
            }

            for ( warpGroupId = 0 ; warpGroupId < (warpCount / LW_PTPC_PRI_SM_ARCH_SUBPARTITION) ; warpGroupId ++ )
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
                    LwU32 smWarpCount = warpCount / LW_PTPC_PRI_SM_ARCH_PARTITION;
                    LwBool warpIsValid;
                    smWarpId = warpGroupId * LW_PTPC_PRI_SM_ARCH_SUBPARTITION + subPartitionId;
                    dprintf("%12d %12d %7d", smWarpId % smWarpCount, subPartitionId, warpGroupId);

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

void grDumpWarpPc_GA102(LwU32 gpcId, LwU32 tpcId, LwU32 repeat, LwBool bAnnotate)
{

    _grDumpTpcWarpPcs(gpcId, tpcId, repeat, bAnnotate);
}

/*!
 * @brief   Function to get the max number of Gpcs
 *
 * @return  Returns LwU32      The max number of Gpcs
 *
 */
LwU32 grGetMaxGpc_GA102(void)
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxTpcPerGpc_GA102()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetNumberPesPerGpc_GA102(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}


/*!
 * Print the CROP exception state for a ROP.
 * @param[in] gpcIdx GPC index
 * @param[in] ropIdx rop index
 * @param[in] hwwEsr Exception State
 */
static void _grPrintCropException_GA102(LwU32 gpcIdx, LwU32 ropIdx, LwU32 hwwEsr)
{
    dprintf("Graphics ROP in GPC Exception: LW_PGRAPH_PRI_GPC%u_ROP%u_CROP_HWW_ESR 0x%x\n", gpcIdx, ropIdx, hwwEsr);

    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_DEAD_TAGS, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_DEAD_TAGS\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_CACHE_HIT_FROM_OTHER_GPC, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_CACHE_HIT_FROM_OTHER_GPC\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_LONG_WAIT_ON_FREE_TAGS, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_LONG_WAIT_ON_FREE_TAGS\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_LONG_WAIT_ON_WRITEACK, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_LONG_WAIT_ON_WRITEACKS\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_NO_FREE_CLEAN_CACHE_ENTRY, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_NO_FREE_CLEAN_CACHE_ENTRY,\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CRD_L2ACK_BUT_WACKFIFO_EMPTY, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_L2ACK_BUT_WACKFIFO_EMPTY\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CPROC_LONG_WAIT_ON_RDAT, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CPROC_LONG_WAIT_ON_RDAT\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _BLEND_FP32_NO_CHANNEL_EN, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: BLEND_FP32_NO_CHANNEL_EN\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _BLEND_SKID, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: BLEND_SKID\n");
    }
    if (hwwEsr & DRF_DEF(_PBE, _PRI_CROP_HWW_ESR, _CWR_NO_EOCP_BW_TILES, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CWR_NO_EOCP_BW_TILES\n");
    }
}

/*!
 * Print the CROP exception state for a GPC.
 * @param[in]  grIdx GR instance index.
 * @param[in] gpcIdx GPC index
 */
static void _grDumpCropExceptionState
(
    LwU32 grIdx,
    LwU32 gpcIdx
)
{
    GR_IO_APERTURE *pRopAperture;
    GR_IO_APERTURE *pGpcAperture;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcIdx),
        return);

    if (REG_FLD_TEST_DRF_DEF(&pGpcAperture->aperture, _PGPC,
            _PRI_GPCCS_GPC_EXCEPTION, _CROP0, _PENDING))
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(pGpcAperture, &pRopAperture, GR_UNIT_TYPE_ROP, 0),
            return);
        _grPrintCropException_GA102(gpcIdx, 0, REG_RD32(&pRopAperture->aperture, LW_PBE_PRI_CROP_HWW_ESR));
    }
    if (REG_FLD_TEST_DRF_DEF(&pGpcAperture->aperture, _PGPC,
            _PRI_GPCCS_GPC_EXCEPTION, _CROP1, _PENDING))
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(pGpcAperture, &pRopAperture, GR_UNIT_TYPE_ROP, 1),
            return);

        _grPrintCropException_GA102(gpcIdx, 1, REG_RD32(&pRopAperture->aperture, LW_PBE_PRI_CROP_HWW_ESR));
    }
}

/*!
 * Print the ZROP exception state for a ROP.
 * @param[in] gpcIdx GPC index
 * @param[in] ropIdx rop index
 * @param[in] hwwEsr Exception State
 */
static void _grPrintZropException_GA102(LwU32 gpcIdx, LwU32 ropIdx, LwU32 hwwEsr)
{
    dprintf("Graphics ROP in GPC Exception: LW_PGRAPH_PRI_GPC%u_ROP%u_ZROP_HWW_ESR 0x%x\n",
        gpcIdx, ropIdx, hwwEsr);

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _DEAD_TAGS, _PRESENT))
    {
        dprintf("Graphics ZROP Exception Type: DEAD_TAGS\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _CACHE_HIT_FROM_OTHER_GPC, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: CACHE_HIT_FROM_OTHER_GPC\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _RDAT_ORPHANS, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: RDAT_ORPHANS\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _EOZP_MISMATCH, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: EOZP_MISMATCH\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _ZWRRAM_ORPHANS, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: ZWRRAM_ORPHANS\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _LONG_WAIT_ON_RDAT, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: LONG_WAIT_ON_RDAT\n");
    }

    if (hwwEsr & DRF_DEF(_PBE, _PRI_ZROP_HWW_ESR, _LONG_WAIT_ON_WRITEACK, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: LONG_WAIT_ON_WRITEACK\n");
    }
}

/*!
 * Print the ZROP exception state for a GPC.
 * @param[in] grIdx GR instance index.
 * @param[in] gpcIdx GPC index
 */
static void _grDumpZropExceptionState
(
    LwU32 grIdx,
    LwU32 gpcIdx
)
{
    GR_IO_APERTURE *pRopAperture;
    GR_IO_APERTURE *pGpcAperture;
    LwU32 ropCount = pGr[indexGpu].grGetNumRopForGpc(gpcIdx, grIdx);
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcIdx),
        return);

    if (REG_FLD_TEST_DRF_DEF(&pGpcAperture->aperture, _PGPC,
            _PRI_GPCCS_GPC_EXCEPTION, _ZROP0, _PENDING))
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(pGpcAperture, &pRopAperture, GR_UNIT_TYPE_ROP, 0),
            return);

        _grPrintZropException_GA102(gpcIdx, 0, REG_RD32(&pRopAperture->aperture, LW_PBE_PRI_ZROP_HWW_ESR));
    }
    if (REG_FLD_TEST_DRF_DEF(&pGpcAperture->aperture, _PGPC,
            _PRI_GPCCS_GPC_EXCEPTION, _ZROP1, _PENDING))
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(pGpcAperture, &pRopAperture, GR_UNIT_TYPE_ROP, 1),
            return);
        _grPrintZropException_GA102(gpcIdx, 1, REG_RD32(&pRopAperture->aperture, LW_PBE_PRI_ZROP_HWW_ESR));
    }
}

/*!
 * Print the ROP exception state for a GPC.
 * @param[in] grIdx GR instance index.
 * @param[in] gpcIdx GPC index
 */
void grDumpRopExceptionState_GA102
(
    LwU32 grIdx,
    LwU32 gpcIdx
)
{
    _grDumpCropExceptionState(grIdx, gpcIdx);
    _grDumpZropExceptionState(grIdx, gpcIdx);
}

void grDumpConsolidatedReportGpc_GA102( LwU32 numActiveGpc, LwU32 grIdx )
{
    LwU32 tpcId;
    LwU32 numActiveTpc;
    LwU32 gpcId;
    LwU32 grStatus;
    LwU32 val;
    LwU32 data32;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    LW_STATUS status;
    GR_IO_APERTURE *pGpcAperture;
    GR_IO_APERTURE *pTpcAperture;

    for (gpcId = 0 ; gpcId < numActiveGpc ; gpcId++)
    {
        LW_ASSERT_OK_OR_ELSE(status,
            GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcId),
            return);


        dprintf("====================\n");
        dprintf("GPC %d\n", gpcId);
        dprintf("====================\n");

        numActiveTpc = pGr[indexGpu].grGetNumTpcForGpc(gpcId, grIdx);
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY0, GA102 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY1, GA102 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY2, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY3, GV100 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_GPC_ACTIVITY4, GA102 );

        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_1, GA102 );
        PRINT_GPC_APERTURE_REG_Z( pGpcAperture, GPCCS_CTXSW_STATUS_GPC_0, GV100 );


        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {

            LW_ASSERT_OK_OR_ELSE(status,
                GR_GET_APERTURE(pGpcAperture, &pTpcAperture, GR_UNIT_TYPE_TPC, tpcId),
                return);

            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, TPCCS_TPC_ACTIVITY0, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, PE_STATUS, GM200 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_VTG_STATUS, GV100 );
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, MPC_PIX_STATUS, GV100 );

            data32 = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING);
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/0\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, TU102 );

            data32 = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING);
            data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
            data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
                LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE1 );
            REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );
            dprintf("====================\n");
            dprintf("GPC/TPC/SM %d/%d/1\n", gpcId, tpcId );
            dprintf("====================\n");
            PRINT_TPC_APERTURE_REG_Z( pTpcAperture, SM_STATUS, TU102 );
        }
    }
}
