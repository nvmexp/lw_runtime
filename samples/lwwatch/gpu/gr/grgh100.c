/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgh100.c
//
//*****************************************************

//
// includes
//
// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/
#include "hopper/gh100/dev_fuse.h"
#include "hopper/gh100/dev_graphics_nobundle.h"
#include "hopper/gh100/dev_gpc_no_tpc.h"
#include "hopper/gh100/hwproject.h"
#include "hopper/gh100/dev_tpc.h"

#include "gr.h"
#include "chip.h"
#include "utils/lwassert.h"
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
grGetUnitApertureInformation_GH100
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

    switch(type)
    {
        case GR_UNIT_TYPE_GR:
            *pUnitBase = LW_PGRAPH_BASE;
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
            return LW_ERR_NOT_SUPPORTED;
    }

    return LW_OK;
}

LwU32 grGetMaxTpcPerGpc_GH100()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetNumberPesPerGpc_GH100(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

LwU32 grGetPesTpcMask_GH100
(
    LwU32 grIdx,
    LwU32 gpcIdx,
    LwU32 pesIdx
)
{
    LwU32 gpcStride;
    LwU32 regAddr;

    if (gpcIdx >= pGr[indexGpu].grGetNumActiveGpc(grIdx))
    {
        dprintf("Illegal GPC index: %d\n", gpcIdx);
        return 0;
    }

    if (pesIdx >= pGr[indexGpu].grGetNumberPesPerGpc())
    {
        dprintf("Illegal PES index : %d\n", pesIdx);
        return 0;
    }

    pGr[indexGpu].grGetUnitApertureInformation(GR_UNIT_TYPE_GPC, NULL, &gpcStride, NULL);
    LW_ASSERT_OR_RETURN(gpcStride != 0, 0);

    regAddr = (LW_PGRAPH_PRI_GPC0_GPM_PD_PES_TPC_ID_MASK(pesIdx) + gpcIdx * gpcStride);
    return GPU_REG_RD32(regAddr);
}

/*!
 *  grTpcExceptionMask_GH100
 *  Return TPC exception mask for identifiying the TPC
 *  which has the GPCCS exception.
 */
LwU32 grTpcExceptionMask_GH100(LwU32 tpcIdx)
{
    // create mask for tpcIdx TPC
    return (1 << (DRF_SHIFT(LW_PGPC_PRI_GPCCS_GPC_EXCEPTION_TPC) + tpcIdx));
}

/*!
 *  grDumpTpcTpccsExceptionState_GH100
 */
void grDumpTpcTpccsExceptionState_GH100(LwU32 gpcIdx, LwU32 tpcIdx)
{
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;
    LwU32 hwwEsr;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcIdx,
            GR_UNIT_TYPE_TPC, tpcIdx),
        return);

    hwwEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TPCCS_TPC_EXCEPTION);

    dprintf("Graphics TPCCS Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_TPCCS_TPC_EXCEPTION 0x%x\n", gpcIdx, tpcIdx, hwwEsr);

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _TPCCS, _PENDING) )
    {
        dprintf("Graphics TPCCS Exception Type: TPCCS\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _SM, _PENDING) )
    {
        dprintf("Graphics TPCCS Exception Type: SM\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _PE, _PENDING) )
    {
        dprintf("Graphics TPCCS Exception Type: PE\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _MPC, _PENDING) )
    {
        dprintf("Graphics TPCCS Exception Type: MPC\n");
    }
}

/*!
 *  grDumpGpccsExceptionState_GH100
 */
void grDumpGpccsExceptionState_GH100(LwU32 gpcCounter, LwU32 tpcCounter)
{
    LW_STATUS status;
    GR_IO_APERTURE *pTpcAperture;
    LwU32 tpccsException = 0;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcCounter,
            GR_UNIT_TYPE_TPC, tpcCounter),
        return);

    tpccsException = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TPCCS_TPC_EXCEPTION);
    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TPCCS_TPC_EXCEPTION:    0x%x\n", gpcCounter, tpcCounter, tpccsException);

    // is it TPCCS exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _TPCCS, _PENDING))
    {
        pGr[indexGpu].grDumpTpcTpccsExceptionState(gpcCounter, tpcCounter);
    }

    // is it SM exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _SM, _PENDING))
    {
        pGr[indexGpu].grDumpTPCSMExceptionState(gpcCounter, tpcCounter);
    }

    // is it PE exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _PE, _PENDING))
    {
        pGr[indexGpu].grDumpTpcPeExceptionState(gpcCounter, tpcCounter);
    }

    // is it MPC exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _MPC, _PENDING))
    {
        pGr[indexGpu].grDumpTpcMpcExceptionState(gpcCounter, tpcCounter);
    }
}

void grDumpTpcInfo_GH100(LwU32 gpcId, LwU32 tpcId)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];
    LwU32 data32;
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

    //
    // These registers hold 3 bit values in ACTIVITY format but those
    // encodings aren't enumerated in the manuals, so priv_dump just
    // prints the numeric values.
    //

    PRINT_TPC_REG_PD(PE_STATUS, gpcId, tpcId);
    DUMP_TPC_APERTURE_REG(pTpcAperture, PE_L2_EVICT_POLICY);
    DUMP_TPC_APERTURE_REG(pTpcAperture, PE_HWW_ESR);

    PRINT_TPC_REG_PD(MPC_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_VTG_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_PIX_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_WLU_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_COMP_STATUS, gpcId, tpcId);
    PRINT_TPC_REG_PD(MPC_COMP_STATUS_2, gpcId, tpcId);

    pGr[indexGpu].grDumpWarpPc( gpcId, tpcId, 1, LW_FALSE );

    DUMP_TPC_APERTURE_REG(pTpcAperture,SM_CONFIG);
    PRINT_TPC_REG_PD(SM_ARCH, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_CACHE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_PRIVATE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_DISP_CTRL, gpcId, tpcId);
    DUMP_TPC_APERTURE_REG(pTpcAperture,SM_MACHINE_ID0);
    // See _GF100 when adding ECC code here
    PRINT_TPC_REG_PD(SM_DEBUG_SFE_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_QUAD_BA_CONTROL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_THROTTLE_CTL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_VST_CTL, gpcId, tpcId);
    PRINT_TPC_REG_PD(SM_POWER_VST_DATA, gpcId, tpcId);

    PRINT_TPC_REG_PD ( SM_BLK_ACTIVITY_PRIV_LEVEL_MASK, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_MIO_CFG2, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_TEXIO_CONTROL, gpcId, tpcId );
    PRINT_TPC_REG_PD ( SM_PM_SAMP_CTRL, gpcId, tpcId );

    // Implementation of bug #1374564
    // TEX PRIs route to PIPE 0 by default
    //
    data32 = REG_RD32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING );
    data32 &= ~DRF_SHIFTMASK( LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL );
    data32 |= DRF_NUM( _PGRAPH, _PRI_GPC0_TPC0_TEX_M_ROUTING, _SEL,
        LW_PGRAPH_PRI_GPC0_TPC0_TEX_M_ROUTING_SEL_PIPE0 );
    REG_WR32( &pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_ROUTING, data32 );

    for (smId = 0; smId < 2; smId++)
    {
        if (smId == 1)
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

        PRINT_TPC_REG_PD(TEX_M_ROUTING, gpcId, tpcId);
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

void grDumpNisoHubStatus_GH100(void)
{
    //LW_PFB_NISO_STATUS
    dprintf("====================\n");
    dprintf("NISO HUB detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PFB_NISO, _DEBUG);
}

LwU32
grGetActiveRopsForGpc_GH100(LwU32 gpcIdx)
{
    LwU32 regVal;
    regVal = GPU_REG_IDX_RD_DRF(_FUSE, _STATUS_OPT_ROP_GPC, gpcIdx, _DATA);
    return ~regVal & (LWBIT32(LW_SCAL_LITTER_NUM_ROP_PER_GPC) - 1);
}


LwU32
grGetActiveCPCsForGpc_GH100(LwU32 gpcIdx)
{
    LwU32 regVal;
    regVal = GPU_REG_IDX_RD_DRF(_FUSE, _STATUS_OPT_CPC_GPC, gpcIdx, _DATA);
    return ~regVal & (LWBIT32(LW_SCAL_LITTER_NUM_CPC_PER_GPC) - 1);
}

LwU32 grGetNumCPCsforGpc_GH100(LwU32 gpcIdx)
{
    return lwPopCount32(grGetActiveCPCsForGpc_GH100(gpcIdx));
}