/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grGK104.c
//
//*****************************************************

//
// includes
//
#include "chip.h"

// HW specific files are in //sw/dev/gpu_drv/chips_a/drivers/common/inc/hwref/
#include "kepler/gk104/dev_graphics_nobundle.h"
#include "kepler/gk104/dev_gpc_no_tpc.h"
#include "kepler/gk104/dev_tpc.h"
#include "kepler/gk104/dev_ctxsw_prog.h"
#include "kepler/gk104/dev_fifo.h"
#include "kepler/gk104/dev_xbar.h"
#include "kepler/gk104/dev_fb.h"
#include "kepler/gk104/hwproject.h"
#include "kepler/gk104/dev_pri_ringmaster.h"
#include "kepler/gk104/dev_pri_ringstation_sys.h"
#include "kepler/gk104/dev_top.h"

#include "inst.h"
#include "print.h"
#include "utils/lwassert.h"
#include "gpuanalyze.h"
#include "gr.h"
#include "heap.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes

// Function declarations

LwU32 grGetMaxTpcPerGpc_GK104()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GK104()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GK104(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

void grDumpIsoHubStatus_GK104(void)
{
    dprintf("ISO HUB status not supported on GK104+\n");
}

// Kepler added a debug bus enable bit.  Since that bit isn't defined
// for Fermi, this code doesn't compile if placed in grgf100.c, so we
// need a KEPLER_and_later version starting with GK104.

void grDumpNisoHubStatus_GK104(void)
{
    LwU32 data32;
    LwU32 i;
    LwU32 niso_debug_config_max;

    //LW_PFB_ISO_STATUS
    dprintf("====================\n");
    dprintf("NISO HUB detailed status\n");
    dprintf("====================\n");
    PRINT_REG_PD(_PFB_NISO, _DEBUG);

    // A mask for the CONFIG field indicates how many configs to
    // iterate on.  See pri_fb.ref in the chip manuals.
    niso_debug_config_max = DRF_MASK( LW_PFB_NISO_PRI_DEBUG1_CONFIG );
    dprintf("Setting _PRI_DEBUG1_CONFIG values (0..%d) and then reading _NISO_DEBUG\n",
        niso_debug_config_max);

    for (i=0; i<= niso_debug_config_max; i++)
    {
        data32 = GPU_REG_RD32(LW_PFB_NISO_PRI_DEBUG1);
        data32 &= ~DRF_SHIFTMASK(LW_PFB_NISO_PRI_DEBUG1_CONFIG);
        data32 |= DRF_NUM(_PFB, _NISO_PRI_DEBUG1, _CONFIG, i);
        data32 |= DRF_NUM(_PFB, _NISO_PRI_DEBUG1, _DEBUG_BUS_EN, LW_PFB_NISO_PRI_DEBUG1_DEBUG_BUS_EN_ON );
        dprintf("Writing LW_PFB_NISO_PRI_DEBUG1_CONFIG=%d\n", i );
        GPU_REG_WR32(LW_PFB_NISO_PRI_DEBUG1, data32);

        PRINT_REG_PD(_PFB, _NISO_PRI_DEBUG1);
        PRINT_REG_PD(_PFB, _NISO_DEBUG);
    }

    // Clear the debug bus enable
    data32 &= ~DRF_SHIFTMASK(LW_PFB_NISO_PRI_DEBUG1_DEBUG_BUS_EN);
    data32 |= DRF_NUM(_PFB, _NISO_PRI_DEBUG1, _DEBUG_BUS_EN, LW_PFB_NISO_PRI_DEBUG1_DEBUG_BUS_EN_OFF );
    GPU_REG_WR32(LW_PFB_NISO_PRI_DEBUG1, data32);
}

void grDumpPpcInfo_GK104(LwU32 gpcId, LwU32 ppcId)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];
    buffer[0] = '\0';

    dprintf("====================\n");
    dprintf("GPC/PPC %d/%d\n", gpcId, ppcId);
    dprintf("====================\n");

    DUMP_PPC_REG(CBM_CONFIG, gpcId, ppcId);
    DUMP_PPC_REG(CBM_DEBUG, gpcId, ppcId);
    PRINT_PPC_REG_PD(CBM_STATUS, gpcId, ppcId);
    DUMP_PPC_REG(CBM_PM, gpcId, ppcId);
    DUMP_PPC_REG(CBM_CG, gpcId, ppcId);
    DUMP_PPC_REG(CBM_CG1, gpcId, ppcId);
    DUMP_PPC_REG(CBM_PG, gpcId, ppcId);
    DUMP_PPC_REG(CBM_PG1, gpcId, ppcId);
    if (! IsGM107orLater())
    {
        PRINT_PPC_REG_PD(CBM_STATUS_CBM, gpcId, ppcId);
    }
    PRINT_PPC_REG_PD(PES_STATUS, gpcId, ppcId);    
    PRINT_PPC_REG_PD(PES_VSC_STATUS, gpcId, ppcId);    
}

void grDumpXbarStatus_GK104( LwU32 grIdx )
{
    LwU32 i;
    LwU32 gpcCount=0, fbpsCount=0;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    dprintf("====================\n");
    dprintf("XBAR detailed status\n");
    dprintf("====================\n");

    pGr[indexGpu].grGetBusInfo(&gpcCount, NULL, &fbpsCount, NULL, NULL, grIdx);

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_MXBAR_CS_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_WXBAR_CS_PRI_GPC%d_GXI_PREG_IDLE_STATUS", i );
        priv_dump(buffer);
    }

    for (i=0; i<gpcCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_GPC%d_GXI_PREG_VTX_ERROR_STATUS", i );
        priv_dump(buffer);
    }

    PRINT_REG_PD(_XBAR_MXBAR,_CS_PRI_SYS0_HXI_PREG_IDLE_STATUS);
    PRINT_REG_PD(_XBAR_CXBAR,_CQ_PRI_SYS0_HXI_PREG_IDLE_STATUS);
    PRINT_REG_PD(_XBAR_WXBAR,_CS_PRI_SYS0_HXI_PREG_IDLE_STATUS);

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_MXBAR_CS_PRI_FBP%d_FXI_PREG_IDLE_STATUS", i);
        priv_dump(buffer);
    }

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_FBP%d_FXI_PREG_IDLE_STATUS", i);
        priv_dump(buffer);
    }

    for (i=0; i<fbpsCount; i++)
    {
        sprintf(buffer, "LW_XBAR_CXBAR_CQ_PRI_FBP%d_FXI_PREG_VTX_ERROR_STATUS", i);
        priv_dump(buffer);
    }
}

LwU32 grGetPesTpcMask_GK104
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

    regAddr = (LW_PGPC_PRI_GPM_PD_PES_TPC_ID_MASK(pesIdx) + gpcIdx * gpcStride);
    return GPU_REG_RD32(regAddr);
}

LwU32 grGetNumPpcForGpc_GK104(LwU32 gpc, LwU32 grIdx)
{
    LwU32 litterNumPesPerGPC = pGr[indexGpu].grGetNumberPesPerGpc();
    LwU32 pesIdx;
    LwU32 ppcPerGpc = 0;

    // check which PES have TPC's assigned to them, if they do we have an active PPC
    for (pesIdx = 0; pesIdx < litterNumPesPerGPC; pesIdx++)
    {
        if (pGr[indexGpu].grGetPesTpcMask(grIdx, gpc, pesIdx))
        {
            ppcPerGpc++;
        }
    }

    return ppcPerGpc;
}

LwU32 grGetNumberPesPerGpc_GK104(void)
{
    return LW_SCAL_LITTER_NUM_PES_PER_GPC;
}

//-----------------------------------------------------
// grCheckSkedStatus_GK104()n
// check status of the SKED unit, record the "error"
// if necessary, consistent with grCheckPipelineStatus()
//-----------------------------------------------------
void grCheckSkedStatus_GK104(char* eng, BOOL bAnalyze)
{
    LwU32 grStatus;

    grStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _SKED, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_SKED_BUSY\n");
        PRINT_REG_PD(_PGRAPH_PRI, _SKED_ACTIVITY);
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_SKED_BUSY\n");
            strcpy(eng, "SKED ()");
        }
    }
}

//-----------------------------------------------------
// grCheckFeConstStatus_GK104()n
// check status of the FECONST unit
// if necessary, consistent with grCheckPipelineStatus()
//-----------------------------------------------------
void grCheckFeConstStatus_GK104(char* eng, BOOL bAnalyze)
{
    LwU32 grStatus;

    grStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _FE_CONST, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_FE_CONST_BUSY\n");
        if (bAnalyze)
        {
            addUnitErr("\t LW_PGRAPH_STATUS_FE_CONST_BUSY\n");
            strcpy(eng, "FE_CONST ()");
        }
    }
}

//
// These registers are at different addresses on Kepler
//
LwU32 grGetNumActiveGpc_GK104(LwU32 grIdx)
{
    LwU32 data32 = GPU_REG_RD32(LW_PPRIV_MASTER_RING_ENUMERATE_RESULTS_GPC);
    return DRF_VAL(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_GPC, _COUNT, data32);
}

LwU32 grGetNumActiveFbp_GK104(void)
{
    LwU32 data32 = GPU_REG_RD32(LW_PPRIV_MASTER_RING_ENUMERATE_RESULTS_FBP);
    return DRF_VAL(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_FBP, _COUNT, data32);
}

/*!
 * @brief A generic constructor for creating sub-apertures of a particular type.
 *
 * The common parts of constructing any aperture involve:
 * 1. Fetching base, stride values.
 * 2. Allocating memory for apertures.
 * 3. Initializing the IO_APERTURE.
 * 4. Updating the parent GPC aperture with new sub-apertures.
 * This Function consolidates the above operations.
 *
 * @param[in] pParent pointer to the parent aperture.
 * @param[in] type Type of apertures to be constructed.
 * @param[in] unitCount number of unicast apertures needed.
 *
 * @return LW_OK upon success.
 *         LW_ERR_ILWALID_ARGUMENT bad values of unit count and type.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures.
 *         LW_ERR* for other failures.
 */
LW_STATUS
grConstructSubApertures_GK104
(
    GR_IO_APERTURE *pParent,
    GR_UNIT_TYPE    type,
    LwU32           unitCount
)
{
    LwU32 base;
    LwU32 stride;
    LwS32 broadcastIdx;
    LwU32 apertureCount;
    LwU32 index;
    GR_IO_APERTURE *pApertures;

    pGr[indexGpu].grGetUnitApertureInformation(type, &base, &stride, &broadcastIdx);

    apertureCount = unitCount + 1; // extra aperture for the broadcast registers

    pApertures = (GR_IO_APERTURE *) malloc(sizeof(*pApertures) * apertureCount);
    LW_ASSERT_OR_RETURN(pApertures != NULL, LW_ERR_NO_MEMORY);

    memset(pApertures, 0, sizeof(*pApertures) * apertureCount);

    pParent->pChildren[type]     = pApertures;
    pParent->sharedIndices[type] = unitCount;
    pParent->unitCounts[type]    = unitCount;

    for (index = 0; index < unitCount; index++)
    {
        LW_ASSERT_OK_OR_RETURN(ioaccessInitIOAperture(&pApertures[index].aperture,
            &pParent->aperture, NULL,
            (base + (index * stride)),
            stride));
        pApertures[index].unitIndex = index;
        pApertures[index].bIsBroadcast = LW_FALSE;
        pApertures[index].pParent = pParent;
    }
    LW_ASSERT_OK_OR_RETURN(ioaccessInitIOAperture(&pApertures[unitCount].aperture,
            &pParent->aperture, NULL,
            (base + (broadcastIdx * stride)),
            stride));
    pApertures[unitCount].unitIndex = unitCount;
    pApertures[unitCount].bIsBroadcast = LW_TRUE;
    pApertures[unitCount].pParent = pParent;
    
    return LW_OK;
}

/*!
 * @brief Constructor for PPC Apertures.
 *
 * @param[in] pGpu pointer to OBJGPU instance.
 * @param[in] pGr pointer to OBJGR instance.
 * @param[in] pGpcAperture pointer to the parent GPC aperture.
 *
 * @return LW_OK upon success.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures.
 *         LW_ERR_NOT_SUPPORTED when PPC count is determined to be 0.
 *         LW_ERR* for other failures.
 */
LW_STATUS
grConstructPpcApertures_GK104
(
    GR_IO_APERTURE  *pGrAperture,
    GR_IO_APERTURE  *pGpcAperture
)
{
    LwU32 ppcCount;
    LwU32 grIdx = pGrAperture->unitIndex;
    LwU32 gpcIdx = 0;

    // ppcCount will the be minimum possible of the unicast units.
    if (pGpcAperture->bIsBroadcast)
    {
        LwU32 minPpcCount = pGr[indexGpu].grGetNumPpcForGpc(0, grIdx);

        for (gpcIdx = 1; gpcIdx < pGrAperture->unitCounts[GR_UNIT_TYPE_GPC]; gpcIdx++)
        {
            minPpcCount = LW_MIN(minPpcCount, pGr[indexGpu].grGetNumPpcForGpc(gpcIdx, grIdx));
        }
        ppcCount = minPpcCount;
    }
    else
    {
        ppcCount = pGr[indexGpu].grGetNumPpcForGpc(pGpcAperture->unitIndex, grIdx);
    }

    // Construct PPC Apertures.
    LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructSubApertures(pGpcAperture, GR_UNIT_TYPE_PPC, ppcCount));

    return LW_OK;
}

/*!
 * @brief Constructor for TPC Apertures.
 *
 * @param[in] pGrAperture pointer to GR_IO_APERTURE instance.
 * @param[in] pAperture pointer to the parent GPC/EGPC aperture.
 * @param[in] bIsExtended flag to indicate if this for extended PRI space.
 *
 * @return LW_OK upon success.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures
 *         LW_ERR_NOT_SUPPORTED when TPC count is found to be 0.
 *         LW_ERR* for other failures.
 */
LW_STATUS
grConstructTpcApertures_GK104
(
    GR_IO_APERTURE *pGrAperture,
    GR_IO_APERTURE *pAperture,
    LwBool          bIsExtended
)
{
    LwU32 tpcCount;
    GR_UNIT_TYPE tpcType = bIsExtended ? GR_UNIT_TYPE_ETPC : GR_UNIT_TYPE_TPC;
    GR_UNIT_TYPE gpcType = bIsExtended ? GR_UNIT_TYPE_EGPC : GR_UNIT_TYPE_GPC;
    LwU32 grIdx = pGrAperture->unitIndex;

    // tpcCount will the be minimum possible for broadcast units.
    if (pAperture->bIsBroadcast)
    {
        LwU32 gpcIdx = 0;
        LwU32 minTpcCount = pGr[indexGpu].grGetNumTpcForGpc(0, grIdx);

        for (gpcIdx = 1; gpcIdx < pGrAperture->unitCounts[gpcType]; gpcIdx++)
        {
            minTpcCount = LW_MIN(minTpcCount,  pGr[indexGpu].grGetNumTpcForGpc(gpcIdx, grIdx));
        }
        tpcCount = minTpcCount;
    }
    else
    {
        tpcCount = pGr[indexGpu].grGetNumTpcForGpc(pAperture->unitIndex, grIdx);
    }

    // Construct TPC Apertures.
    LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructSubApertures(pAperture, tpcType, tpcCount));

    return LW_OK;
}

/*!
 * @brief Helper used to fetch an IO aperture based on type and index
 * information of its parent apertures.
 *
 * @param[in]  pApertureIn   pointer to Input Aperture (NULL translates to top-level GR Aperture).
 * @param[out] ppApertureOut pointer to Output Aperture.
 * @param[in]  pTypeIndex    pointer to Array of type-Index pairs, each pair is (GR_UNIT_TYPE*, index)
 * @param[in]  count         Number of elements in the typeIndex array.
 */
LW_STATUS
grGetAperture_GK104
(
    GR_IO_APERTURE *pApertureIn,
    GR_IO_APERTURE **ppApertureOut,
    LwU32          *pTypeIndex,
    LwU32           count
)
{
    LwU32 i;
    GR_UNIT_TYPE type;
    LwU32 apIndex;

    if (pApertureIn == NULL || ppApertureOut == NULL || pTypeIndex == NULL) {
        dprintf("lw: %s: %d: Null pointer.\n", __FUNCTION__, __LINE__);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    // check if count is even
    LW_ASSERT_OR_RETURN((~count) & LWBIT32(0), LW_ERR_ILWALID_ARGUMENT);

    // Array is composed of type-index pairs, so increment by 2 after operating on each pair.
    for (i = 0; i < count; i += 2)
    {
        type = (GR_UNIT_TYPE)pTypeIndex[i]; // type should be at even index.

        LW_ASSERT_OR_RETURN(type < GR_UNIT_TYPE_COUNT, LW_ERR_ILWALID_ARGUMENT);
        LW_ASSERT_OR_RETURN(pApertureIn->pChildren[type] != NULL, LW_ERR_ILWALID_ARGUMENT);

        apIndex = pTypeIndex[i + 1];
        if (apIndex == GR_BROADCAST)
        {
            apIndex = pApertureIn->sharedIndices[type];
        }
        else
        {
            LW_ASSERT_OR_RETURN(apIndex < pApertureIn->unitCounts[type], LW_ERR_ILWALID_ARGUMENT);
        }

        pApertureIn = &pApertureIn->pChildren[type][apIndex];
    }

    LW_ASSERT_OR_RETURN(pApertureIn != NULL, LW_ERR_OBJECT_NOT_FOUND);

    *ppApertureOut = pApertureIn;
    return LW_OK;
}

/*!
 * @brief Helper for Initializing GPC/EGPC Apertures.
 *
 * @param[in] pGrAperture pointer to the parent GR aperture.
 * @param[in] pGpcAperture pointer to the parent GPC aperture.
 * @param[in] apType GPC or EGPC aperture type
 *
 * @return LW_OK upon success.
 *         LW_ERR_NO_MEMORY Out of memory when allocating Apertures.
 *         LW_ERR_NOT_SUPPORTED when a subunit count is determined to be 0.
 *         LW_ERR* for other failures.
 */
static LW_STATUS
_grInitGpcAperture
(
    GR_IO_APERTURE *pGrAperture,
    GR_IO_APERTURE *pGpcAperture,
    GR_UNIT_TYPE    apType
)
{
    if (apType == GR_UNIT_TYPE_GPC)
    {
        // Construct ROP Apertures.
        LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructRopAperturesStruct(pGrAperture, pGpcAperture));

        // Construct PPC apertures.
        LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructPpcApertures(pGrAperture, pGpcAperture));

        // Construct TPC apertures.
        LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructTpcApertures(pGrAperture, pGpcAperture, LW_FALSE));
    }
    else if (apType == GR_UNIT_TYPE_EGPC)
    {
        // Construct ETPC apertures.
        LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructTpcApertures(pGrAperture, pGpcAperture, LW_TRUE));
    }

    return LW_OK;
}

/*!
 * @brief A constructor for GPC/EGPC Apertures.
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
grConstructGpcApertures_GK104
(
    GR_IO_APERTURE *pGrAperture,
    LwBool  bIsExtended
)
{
    LwU32 gpcCount;
    LwU32 gpcIndex;
    GR_IO_APERTURE *pAperture = NULL;
    GR_UNIT_TYPE gpcType = bIsExtended ? GR_UNIT_TYPE_EGPC : GR_UNIT_TYPE_GPC;

    gpcCount = pGr[indexGpu].grGetNumActiveGpc(pGrAperture->unitIndex);

    // Construct GPC Apertures.
    LW_ASSERT_OK_OR_RETURN(pGr[indexGpu].grConstructSubApertures(pGrAperture, gpcType, gpcCount));

    for (gpcIndex = 0; gpcIndex < gpcCount; gpcIndex++)
    {
        // Initialize unicast GPC Apertures.
        LW_ASSERT_OK_OR_RETURN(GR_GET_APERTURE(pGrAperture, &pAperture, gpcType, gpcIndex));
        LW_ASSERT_OK_OR_RETURN(_grInitGpcAperture(pGrAperture, pAperture, gpcType));
    }

    // Initialize broadcast GPC Aperture.
    LW_ASSERT_OK_OR_RETURN(GR_GET_APERTURE(pGrAperture, &pAperture, gpcType, GR_BROADCAST));
    LW_ASSERT_OK_OR_RETURN(_grInitGpcAperture(pGrAperture, pAperture, gpcType));

    return LW_OK;
}

/*!
 * @brief Destroy all resources held by GR IO Apertures.
 *
 * @param[in] pGrAperture  - pointer to GR Aperture.
 */
void
grDestroyIoApertures_GK104(GR_IO_APERTURE *pGrAperture)
{
    GR_IO_APERTURE *pGpcApertures;
    GR_IO_APERTURE *pEgpcApertures;
    LwU32 gpcIndex;
    LwU32 gpcCount;
    GR_UNIT_TYPE apType;
    LwU32 gpcApertureCount;
    LwU32 grIdx = pGrAperture->unitIndex;

    gpcCount = pGrAperture->unitCounts[GR_UNIT_TYPE_GPC];

    gpcApertureCount = gpcCount + 1;

    pGpcApertures = pGrAperture->pChildren[GR_UNIT_TYPE_GPC];
    pEgpcApertures = pGrAperture->pChildren[GR_UNIT_TYPE_EGPC];

    for (gpcIndex = 0; gpcIndex < gpcApertureCount; gpcIndex++)
    {
        if (pGpcApertures != NULL)
        {
            for (apType = GR_UNIT_TYPE_GPU_DEVICE; apType < GR_UNIT_TYPE_COUNT; apType++)
            {
                free(pGpcApertures[gpcIndex].pChildren[apType]);
                pGpcApertures[gpcIndex].pChildren[apType] = NULL;
            }
        }

        if (pEgpcApertures != NULL)
        {
            free(pEgpcApertures[gpcIndex].pChildren[GR_UNIT_TYPE_ETPC]);
            pEgpcApertures[gpcIndex].pChildren[GR_UNIT_TYPE_ETPC] = NULL;
        }
    }

    free(pGpcApertures);
    pGrAperture->pChildren[GR_UNIT_TYPE_GPC]  = NULL;
    pGrAperture->unitCounts[GR_UNIT_TYPE_GPC] = 0;

    free(pEgpcApertures);
    pGrAperture->pChildren[GR_UNIT_TYPE_EGPC]  = NULL;
    pGrAperture->unitCounts[GR_UNIT_TYPE_EGPC] = 0;
}

/*!
 * @brief Create IO_APERTUREs for GR and its sub-units for Hopper architecture.
 *
 * @param[in] pGpuAperture  pointer to GPU IO_APERTURE instance.
 * @param[in] pGrAperture   pointer to GR GR_IO_APERTURE instance.
 *
 * @returns LW_STATUS LW_OK upon success.
 *                    LW_ERR* otherwise.
 */
LW_STATUS
grConstructIoApertures_GK104(GR_IO_APERTURE *pGrAperture, IO_APERTURE *gpuAperture)
{
    LW_STATUS status = LW_OK;

    //
    // The broadcast index can be negative because the broadcast range can be
    // above the unicast range in PRI space. For example,
    // If GPC unicast base = 0x50000 with stride = 0x10000
    // With gpcBCIndex = -2 meaning that the broadcast range base is:
    // 0x50000 + (-2) * 0x10000 = 0x30000
    //
    LwU32 grBase;
    LwU32 grStride;
    LwS32 unused;

    pGr[indexGpu].grGetUnitApertureInformation(GR_UNIT_TYPE_GR, &grBase, &grStride, &unused);
    if (grBase == 0)
    {
        // 0-based manuals is not supported yet for the architecture.
        return LW_OK;
    }

    // initialize GR Aperture.
    LW_ASSERT_OK_OR_GOTO(status,
       ioaccessInitIOAperture(&(pGrAperture->aperture), gpuAperture, NULL, grBase, grStride),
       done);

    // Create apertures for GPC registers.
    LW_ASSERT_OK_OR_GOTO(status,
        pGr[indexGpu].grConstructGpcApertures(pGrAperture, LW_FALSE),
        done);

    // Create apertures for EGPC registers.
    LW_ASSERT_OK_OR_GOTO(status,
        pGr[indexGpu].grConstructEgpcAperturesStruct(pGrAperture),
        done);
done:
    if (status != LW_OK)
    {
        pGr[indexGpu].grDestroyIoApertures(pGrAperture);
    }

    return status;
}

// LW_PFIFO_ENGINE_STATUS() has a different stride starting on Kepler
void grPrintEngineGraphicsStatus_GK104(void)
{
    char buffer[GR_REG_NAME_BUFFER_LEN];

    // A little tricky: print address of this reg as a string in order
    // to use priv_dump, which will print all of the fields
    // automatically.  It won't print the fact that _ENGINE_GRAPHICS
    // is the register being shown (most likely the value will be 0),
    // so print that beforehand to make it clear.  priv_dump will
    // recognize "LW_PFIFO_ENGINE_STATUS(0)" as an address, but not
    // "LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS)".  Using them
    // this way expands and checks the values at compile time.
    dprintf("LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS):\n");
    sprintf( buffer, "0x%08x", LW_PFIFO_ENGINE_STATUS(LW_PFIFO_ENGINE_GRAPHICS) );
    priv_dump( buffer );
}

/*!
 * @brief Provides the caller with information about a particular type of GR Aperture.
 *
 * Output parameters may be NULL if that particular aperture information is not
 * required.
 *
 * @param[in]  type        type of the Aperture, GR_UNIT_TYPE* macros defined in grunits.h
 * @param[out] pUnitBase   Base address for the first unit Aperture of its kind
 * @param[out] pUnitStride Stride length for the scalable unit
 * @param[out] pUnitBCIdx  Signed index for a broadcast Aperture, relative to Base*
 *
 * @return LW_STATUS LW_OK upon success
 *                   LW_ERR_NOT_SUPPORTED for an unknown aperture type for this Arch.
 */
LW_STATUS
grGetUnitApertureInformation_GK104
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

    switch (type)
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

        default:
            return LW_ERR_NOT_SUPPORTED;
    }
    return LW_OK;
}

/*----------------------------------------------------------------------------
 * static void
 * grDumpConsolidatedReport_GK104()
 *      Print a tightly formatted report of chip state, allowing quick
 *      review by SW teams when determining if a bug needs to be
 *      looked at by HW, and by HW teams when determining where to
 *      look next.  HALified because new bit fields for Kepler won't
 *      compile if this is placed in grgf100.c, and fields removed
 *      after Fermi won't compile in grgk104.c (even if guarded).
 *
 * Return Value --
 *      void.
 *
 *----------------------------------------------------------------------------
 */

void grDumpConsolidatedReport_GK104( LwU32 grIdx )
{
    LwU32   grStatus, fbpCount, gpcCount, val, pgraphStatus;
    LwU32   act0, act1, act2, act4;
    char    regName[GR_REG_NAME_BUFFER_LEN];
    char    buffer[GR_REG_NAME_BUFFER_LEN];

    PRINT_REG2_Z(_PGRAPH, _STATUS, GK104);
    PRINT_REG_Z(_PGRAPH_GRFIFO, _STATUS );
    PRINT_REG2_Z(_PGRAPH, _INTR, GF100 );
    PRINT_REG_Z(_PGRAPH, _PRI_FECS_HOST_INT_STATUS );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_FE_0, GK104 );
    PRINT_REG2_Z(_PGRAPH, _PRI_FECS_CTXSW_STATUS_1, GK104 );
    PRINT_REG2_Z(_PGRAPH, _EXCEPTION, GK104 );

    pGr[indexGpu].grGetBusInfo( &gpcCount, NULL, &fbpCount, NULL, NULL, grIdx );
    pgraphStatus = GPU_REG_RD32(LW_PGRAPH_STATUS);
    act0 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY0);
    act1 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY1);
    act2 = GPU_REG_RD32(LW_PGRAPH_ACTIVITY2);
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
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PMA, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMA, act1),
                        "PMA");
    }
    if (pgraphStatus & DRF_DEF(_PGRAPH, _STATUS, _PMMSYS, _BUSY))
    {
        pGr[indexGpu].grPrintGpuUnitStatus(DRF_VAL(_PGRAPH, _ACTIVITY1, _PMMSYS, act1),
                        "PMMSYS");
    }
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

void grDumpTPCL1CExceptionState_GK104(LwU32 gpcCounter, LwU32 tpcCounter)
{
    LwU32 hwwEsr;
    LwU32 hwwEsrAddr;
    LwU32 hwwEsrReq;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;
    dprintf(" _TPC%d_TPCCS_TPC_EXCEPTION_L1C_PENDING\n", tpcCounter);

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcCounter,
            GR_UNIT_TYPE_TPC, tpcCounter),
        return);

    hwwEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_L1C_HWW_ESR);
    hwwEsrAddr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_L1C_HWW_ESR_ADDR);
    hwwEsrReq = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_L1C_HWW_ESR_REQ );

    dprintf("Graphics L1 Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_L1C_HWW_ESR 0x%x\n", gpcCounter, tpcCounter, hwwEsr);
    dprintf("Graphics L1 Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_L1C_HWW_ESR_ADDR 0x%x\n", gpcCounter, tpcCounter, hwwEsrAddr);
    dprintf("Graphics L1 Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_L1C_HWW_ESR_REQ 0x%x\n", gpcCounter, tpcCounter, hwwEsrReq);

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_L1C_HWW_ESR, _LOCAL_SZ, _PENDING))
    {
        dprintf("Graphics L1 Exception Type: LOCAL_SZ\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_L1C_HWW_ESR, _ECC_ERR, _PENDING))
    {
        dprintf("Graphics L1 Exception Type: ECC_ERR\n");
    }

    // Print the opcode
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_GLOBAL_STORE)
    {
        dprintf("Graphics L1 OPCODE: GLOBAL_STORE\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_GLOBAL_LOAD)
    {
        dprintf("Graphics L1 OPCODE: GLOBAL_LOAD\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_LOCAL_STORE)
    {
        dprintf("Graphics L1 OPCODE: LOCAL_STORE\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_LOCAL_LOAD)
    {
        dprintf("Graphics L1 OPCODE: LOCAL_LOAD\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_MEMBAR)
    {
        dprintf("Graphics L1 OPCODE: MEMBAR\n");
    }

    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_CACHE_CCTL)
    {
        dprintf("Graphics L1 OPCODE: CACHE_CCTL\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_STACK_STORE)
    {
        dprintf("Graphics L1 OPCODE: STACK_STORE\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_STACK_LOAD)
    {
        dprintf("Graphics L1 OPCODE: STACK_LOAD\n");
    }
    if (DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _OPCODE, hwwEsrReq) == LW_PTPC_PRI_L1C_HWW_ESR_REQ_OPCODE_FLUSH)
    {
        dprintf("Graphics L1 OPCODE: _FLUSH\n");
    }

    dprintf("Graphics L1 Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_L1C_HWW_ESR_REQ_SM_WARP_ID 0x%x\n",
         gpcCounter, tpcCounter, DRF_VAL(_PTPC, _PRI_L1C_HWW_ESR_REQ, _SM_WARP_ID, hwwEsrReq));

}

void grDumpConsolidatedReportGpc_GK104( LwU32 numActiveGpc, LwU32 grIdx )
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
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY2, GK104 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_GPC_ACTIVITY3, GF100 );

        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_1, GK104 );
        PRINT_GPC_REG_Z( gpcId, GPCCS_CTXSW_STATUS_GPC_0, GF100 );

        for (tpcId = 0 ; tpcId < numActiveTpc ; tpcId++)
        {
            dprintf("====================\n");
            dprintf("GPC/TPC %d/%d\n", gpcId, tpcId );
            dprintf("====================\n");

            PRINT_TPC_REG_Z( gpcId, tpcId, TPCCS_TPC_ACTIVITY0, GK104 );
            PRINT_TPC_REG_Z( gpcId, tpcId, PE_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_VTG_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, MPC_PIX_STATUS, GF100 );
            PRINT_TPC_REG_Z( gpcId, tpcId, SM_INFO_SUBUNIT_STATUS, GK104 );
        }
    }
}

void grDumpBeCropExceptionState_GK104(LwU32 beCounter, LwU32 beOffset)
{
    LwU32 hwwEsr;
    dprintf(" BE_EXCEPTION_CROP_PENDING\n");
    hwwEsr = GPU_REG_RD32(LW_PGRAPH_PRI_BE0_CROP_HWW_ESR + beOffset);

    dprintf("Graphics BE Exception: LW_PGRAPH_PRI_BE%d_CROP_HWW_ESR 0x%x\n", beCounter, hwwEsr);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_DEAD_TAGS, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_DEAD_TAGS\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_CACHE_HIT_FROM_OTHER_GPC, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_CACHE_HIT_FROM_OTHER_GPC\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_LONG_WAIT_ON_FREE_TAGS, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_LONG_WAIT_ON_FREE_TAGS\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_LONG_WAIT_ON_WRITEACK, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_LONG_WAIT_ON_WRITEACKS\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_NO_FREE_CLEAN_CACHE_ENTRY, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_NO_FREE_CLEAN_CACHE_ENTRY,\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CRD_L2ACK_BUT_WACKFIFO_EMPTY, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CRD_L2ACK_BUT_WACKFIFO_EMPTY\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CPROC_LONG_WAIT_ON_RDAT, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CPROC_LONG_WAIT_ON_RDAT\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _BLEND_FP32_NO_CHANNEL_EN, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: BLEND_FP32_NO_CHANNEL_EN\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _BLEND_SKID, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: BLEND_SKID\n");
    }
    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_CROP_HWW_ESR, _CWR_NO_EOCP_BW_TILES, _VIOL_PENDING))
    {
        dprintf("Graphics CROP Exception Type: CWR_NO_EOCP_BW_TILES\n");
    }
}

void grDumpBeZropExceptionState_GK104(LwU32 beCounter, LwU32 beOffset)
{
    LwU32 hwwEsr;
    dprintf(" BE_EXCEPTION_ZROP_PENDING\n");

    hwwEsr = GPU_REG_RD32(LW_PGRAPH_PRI_BE0_ZROP_HWW_ESR + beOffset);

    dprintf("Graphics BE Exception: LW_PGRAPH_PRI_BE%d_ZROP_HWW_ESR 0x%x\n", beCounter, hwwEsr);

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _DEAD_TAGS, _PRESENT))
    {
        dprintf("Graphics ZROP Exception Type: DEAD_TAGS\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _CACHE_HIT_FROM_OTHER_GPC, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: CACHE_HIT_FROM_OTHER_GPC\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _RDAT_ORPHANS, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: RDAT_ORPHANS\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _EOZP_MISMATCH, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: EOZP_MISMATCH\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _ZWRRAM_ORPHANS, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: ZWRRAM_ORPHANS\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _LONG_WAIT_ON_RDAT, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: LONG_WAIT_ON_RDAT\n");
    }

    if (hwwEsr & DRF_DEF(_PGRAPH, _PRI_BE0_ZROP_HWW_ESR, _LONG_WAIT_ON_WRITEACK, _TRUE))
    {
        dprintf("Graphics ZROP Exception Type: LONG_WAIT_ON_WRITEACK\n");
    }
}

void grCheckBePipelineStatus_GK104
(
    char* eng,
    LwU32 grStatus,
    LwU32 numActiveFbps
)
{
    LwU32 fbpId;
    char buffer[GR_REG_NAME_BUFFER_LEN];

    //(cbar, xbar, crop[*], zrop[*])
    // BE_BUSY is a rollup of Back End context switched status from BECS, one per FBP
    // So if BE_BUSY is true, go through each FBP and report BECS state.
    if (grStatus & DRF_DEF(_PGRAPH, _STATUS, _BE, _BUSY))
    {
        dprintf("LW_PGRAPH_STATUS_BE_BUSY\n");
        if (eng != NULL) {
            addUnitErr("\t LW_PGRAPH_STATUS_BE_BUSY\n");
            strcpy(eng, "BE (cbar, xbar, crop[*], zrop[*])");
        }

        // Dump per BE sub-unit ACTIVITY information
        for (fbpId = 0 ; fbpId < numActiveFbps ; fbpId++ )
        {
            PRINT_BE_REG_PD(BECS_BE_ACTIVITY*, fbpId);
        }
    }
}

void grDumpPgraphBeExceptionEnState_GK104
(
    LwU32 regEn
)
{
    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_BE, regEn) == LW_PGRAPH_EXCEPTION_EN_BE_DISABLED)
    {
        dprintf("         _EXCEPTION_EN_BE_DISABLED\n");
    }

}
void grDumpPgraphBeExceptionsState_GK104
(
    LwU32 regExc,
    LwU32 fbpsCount
)
{
    LwU32 regBeExcTwo = 0;
    LwU32 beCounter = 0;
    LwU32 beOffset = 0;
    LwU32 beExceptionTwo = 0;
    LwU32 becsException = 0;


    if (DRF_VAL(_PGRAPH, _EXCEPTION,_BE, regExc) == LW_PGRAPH_EXCEPTION_BE_PENDING)
    {
        // read exception code reg for BE
        regBeExcTwo = GPU_REG_RD32(LW_PGRAPH_EXCEPTION2);
        dprintf("LW_PGRAPH_EXCEPTION2:    0x%x\n", regBeExcTwo);

        for (beCounter = 0; beCounter < fbpsCount; beCounter++)
        {
            // get offset for this BE
            beOffset = LW_ROP_PRI_STRIDE * beCounter;

            // check if this particular BE has exception.
            if (beExceptionTwo & BIT(beCounter))
            {
                // BE (i) has an exception
                becsException = GPU_REG_RD32(LW_PGRAPH_PRI_BE0_BECS_BE_EXCEPTION + beOffset);
                dprintf("LW_PGRAPH_PRI_BE%d_BECS_BE_EXCEPTION:    0x%x\n", beOffset, becsException);

                // is it CROP exception?
                if (becsException & DRF_DEF(_PGRAPH, _PRI_BE0_BECS_BE_EXCEPTION, _CROP, _PENDING))
                {
                    pGr[indexGpu].grDumpBeCropExceptionState( beCounter, beOffset);
                }

                // is it ZROP exception?
                if (becsException & DRF_DEF(_PGRAPH, _PRI_BE0_BECS_BE_EXCEPTION, _ZROP, _PENDING))
                {
                    pGr[indexGpu].grDumpBeZropExceptionState(beCounter, beOffset);
                }
            }
        }
    }
}
void grDumpSmLrfEccState_GK104
(
    LwU32 gpcIdx,
    LwU32 tpcIdx
)
{
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;
    LwU32 eccStatus;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcIdx,
            GR_UNIT_TYPE_TPC, tpcIdx),
        return);

    eccStatus = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_LRF_ECC_STATUS);

    if (eccStatus)
    {
        if (eccStatus & DRF_DEF(_PTPC, _PRI_SM_LRF_ECC_STATUS, _SINGLE_ERR_DETECTED, _PENDING))
        {
            dprintf("Graphics SM ECC Single Bit Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_PRI_GPC0_TPC0_SM_LRF_ECC_STATUS_SINGLE_ERR_DETECTED\n",
                        gpcIdx, tpcIdx);
        }

        if (eccStatus & DRF_DEF(_PTPC, _PRI_SM_LRF_ECC_STATUS, _DOUBLE_ERR_DETECTED, _PENDING))
        {
            dprintf("Graphics SM ECC Double Bit Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_PRI_GPC0_TPC0_SM_LRF_ECC_STATUS_DOUBLE_ERR_DETECTED\n",
                        gpcIdx, tpcIdx);
        }
    }
}

LwU32 grGetSmHwwEsrWarpId_GK104(LwU32 hwwWarpEsr)
{
    return DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _WARP_ID, hwwWarpEsr);
}

void grDumpTPCSMExceptionState_GK104
(
    LwU32 gpcIdx,
    LwU32 tpcIdx
)
{
    LwU32 hwwWarpEsr, hwwWarpRptMask, hwwGlbEsr, hwwGlbRptMask;
    LwU32 eccStatus;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcIdx,
            GR_UNIT_TYPE_TPC, tpcIdx),
        return);

    dprintf(" _TPC0_TPCCS_TPC_EXCEPTION_SM_PENDING\n");

    hwwWarpEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_HWW_WARP_ESR);
    hwwWarpRptMask = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_HWW_WARP_ESR_REPORT_MASK);

    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR :0x%x", gpcIdx, tpcIdx, hwwWarpEsr);
    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_REPORT_MASK :0x%x\n", gpcIdx, tpcIdx, hwwWarpRptMask);

    dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR 0x%x\n", gpcIdx, tpcIdx, hwwWarpEsr);
    dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_REPORT_MASK 0x%x\n", gpcIdx, tpcIdx, hwwWarpRptMask);
    dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_WARP_ID = %d\n", gpcIdx, tpcIdx,
            pGr[indexGpu].grGetSmHwwEsrWarpId(hwwWarpEsr));

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_NONE)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_NONE\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_STACK_ERROR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_STACK_ERROR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_API_STACK_ERROR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_API_STACK_ERROR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_RET_EMPTY_STACK_ERROR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_RET_EMPTY_STACK_ERROR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_PC_WRAP)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_PC_WRAP\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_MISALIGNED_PC)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_MISALIGNED_PC\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_PC_OVERFLOW)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_PC_OVERFLOW\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_MISALIGNED_IMMC_ADDR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_MISALIGNED_IMMC_ADDR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_MISALIGNED_REG)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_MISALIGNED_REG)\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_ENCODING)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_ENCODING\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILLEGAL_SPH_INSTR_COMBO)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILLEGAL_SPH_INSTR_COMBO\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_OOR_REG)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_OOR_REG\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_OOR_ADDR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_OOR_ADDR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_MISALIGNED_ADDR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_MISALIGNED_ADDR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILWALID_ADDR_SPACE)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILWALID_ADDR_SPACE\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM2)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILLEGAL_INSTR_PARAM2\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR_LDC)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_ILWALID_CONST_ADDR_LDC\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_GEOMETRY_SM_ERROR)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_GEOMETRY_SM_ERROR\n", gpcIdx, tpcIdx);
    }

    if (DRF_VAL(_PTPC, _PRI_SM_HWW_WARP_ESR, _ERROR, hwwWarpEsr) == LW_PTPC_PRI_SM_HWW_WARP_ESR_ERROR_DIVERGENT)
    {
        dprintf("Graphics SM HWW Warp Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_WARP_ESR_ERROR_DIVERGENT\n", gpcIdx, tpcIdx);
    }

    // Handle GLOBAL Error
    hwwGlbEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_HWW_GLOBAL_ESR);
    hwwGlbRptMask = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_HWW_GLOBAL_ESR_REPORT_MASK);

    dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR 0x%x\n", gpcIdx, tpcIdx, hwwGlbEsr);
    dprintf("Graphics SM Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR_REPORT_MASK 0x%x\n", gpcIdx, tpcIdx, hwwGlbRptMask);

    if (hwwGlbEsr & DRF_DEF(_PTPC, _PRI_SM_HWW_GLOBAL_ESR, _SM_TO_SM_FAULT, _PENDING))
    {
        dprintf("Graphics SM HWW Global Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR_SM_TO_SM_FAULT\n", gpcIdx, tpcIdx);
    }

    if (hwwGlbEsr & DRF_DEF(_PTPC, _PRI_SM_HWW_GLOBAL_ESR, _L1_ERROR, _PENDING))
    {
        dprintf("Graphics SM HWW Global Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR_L1_ERROR\n", gpcIdx, tpcIdx);
    }

    if (hwwGlbEsr & DRF_DEF(_PTPC, _PRI_SM_HWW_GLOBAL_ESR, _MULTIPLE_WARP_ERRORS, _PENDING))
    {
        dprintf("Graphics SM HWW Global Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR_PHYSICAL_MULTIPLE_WARP_ERRORS\n", gpcIdx, tpcIdx);
    }

    if (hwwGlbEsr & DRF_DEF(_PTPC, _PRI_SM_HWW_GLOBAL_ESR, _PHYSICAL_STACK_OVERFLOW_ERROR, _PENDING))
    {
        dprintf("Graphics SM HWW Global Esr Error - LW_PGRAPH_PRI_GPC%d_TPC%d_SM_HWW_GLOBAL_ESR_PHYSICAL_STACK_OVERFLOW_ERROR\n", gpcIdx, tpcIdx);
    }


    eccStatus = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_SM_LRF_ECC_STATUS);

    if (eccStatus)
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_SM_LRF_ECC_STATUS :0x%x\n", gpcIdx, tpcIdx, eccStatus);
        pGr[indexGpu].grDumpSmLrfEccState(gpcIdx, tpcIdx);
    }
}

LwU32 grGetPteKindFromPropHwwEsr_GK104(LwU32 regVal)
{
    return DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _PTE_KIND, regVal);    
}

/*!
 * Dump PROP exception state.
 * @param gpcIdx : GPC index
 */
void grDumpPropExceptionState_GK104(LwU32 gpcIdx)
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

    dprintf("LW_PGRAPH_PRI_GPC%d_PROP_HWW_ESR:0x%x", gpcIdx, hwwEsr);
    dprintf("LW_PGRAPH_PRI_GPC%d_PROP_HWW_ESR_COORD:0x%x", gpcIdx, esrCoord);
    dprintf("LW_PGRAPH_PRI_GPC%d_PROP_HWW_ESR_FORMAT:0x%x", gpcIdx, esrFormat);
    dprintf("LW_PGRAPH_PRI_GPC%d_PROP_HWW_ESR_STATE:0x%x\n", gpcIdx, esrState);

    dprintf(" Graphics PROP Exception: LW_PGRAPH_PRI_GPC%d_PROP_HWW_ESR 0x%x\n", gpcIdx, hwwEsr);

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _PITCH_2D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_PITCH_2D_VIOL\n", gpcIdx);

        dprintf(" Graphics PROP Exception Type:    PROP PITCH_2D Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DST FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _DST, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _PITCH_3D_VIOL, _PENDING))
    {
       dprintf("           _GPC%d_PROP_HWW_ESR_PITCH_3D_VIOL\n", gpcIdx);
       dprintf(" Graphics PROP Exception Type:    PROP PITCH_3D Violation on GPC %d\n", gpcIdx);

        dprintf(
                    "PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(
                    "PROP%d:    COLOR FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _CT, esrFormat));
        dprintf(
                    "PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));
    }

    if ((hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _WIDTH_3D_ZT_VIOL, _PENDING)) ||
        (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _HEIGHT_3D_ZT_VIOL, _PENDING)))
    {
        if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _WIDTH_3D_ZT_VIOL, _PENDING))
        {
            dprintf("           _GPC%d_PROP_HWW_ESR_WIDTH_3D_ZT_VIOL\n", gpcIdx);
            dprintf(" Graphics PROP Exception Type:    PROP WIDTH_3D ZT Violation on GPC %d\n", gpcIdx);
        }
        else
        {
            dprintf("           _GPC%d_PROP_HWW_ESR_HEIGHT_3D_ZT_VIOL\n", gpcIdx);
            dprintf(" Graphics PROP Exception Type:    PROP WIDTH_3D ZT Violation on GPC %d\n", gpcIdx);
        }

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DEPTH FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _ZT, esrFormat));

        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));
    }

    if ((hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _HEIGHT_3D_CT_VIOL, _PENDING)) ||
        (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _WIDTH_3D_CT_VIOL, _PENDING)))
    {
        if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _HEIGHT_3D_CT_VIOL, _PENDING))
        {
            dprintf("           _GPC%d_PROP_HWW_ESR_HEIGHT_3D_CT_VIOL\n", gpcIdx);
            dprintf(" Graphics PROP Exception Type:    PROP HEIGHT_3D_CT Violation on GPC %d\n", gpcIdx);
        }
        else
        {
           dprintf("           _GPC%d_PROP_HWW_ESR_WIDTH_3D_CT_VIOL\n", gpcIdx);
           dprintf(" Graphics PROP Exception Type:    PROP WIDTH_CT Violation on GPC %d\n", gpcIdx);
        }

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    COLOR FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _CT, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_2D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_2D_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 2D KIND Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DST FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _DST, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_3D_ZT_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_3D_ZT_VIOL\n", gpcIdx);

        dprintf(" Graphics PROP Exception Type:    PROP 3D-Z KIND Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DEPTH FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _ZT, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_3D_CT_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_3D_CT_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 3D-C KIND Violation on GPC %d\n", gpcIdx);

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

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _MEMLAYOUT_2D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_MEMLAYOUT_2D_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 2D MEMLAYOUT Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DST FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _DST, esrFormat));

        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _MEMLAYOUT_3D_CT_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_MEMLAYOUT_3D_CT_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 3D-C MEMLAYOUT Violation on GPC %d\n", gpcIdx);

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

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _ZBLEND_2D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_ZBLEND_2D_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 2D ZBLEND Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DST FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _DST, esrFormat));

        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _ZBLEND_3D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_ZBLEND_3D_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 3D ZBLEND Violation on GPC %d\n", gpcIdx);

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

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_SYSMEM_2D_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_SYSMEM_2D_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 2D SYSMEM KIND Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DST FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _DST, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_SYSMEM_3D_ZT_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_SYSMEM_3D_ZT_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 3D-Z SYSMEM KIND Violation on GPC %d\n", gpcIdx);

        dprintf(" PROP%d:    X COORD: 0x%x, Y COORD: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _X, esrCoord),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_COORD, _Y, esrCoord));

        dprintf(" PROP%d:    DEPTH FORMAT: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _ZT, esrFormat));
        dprintf(" PROP%d:    AA_SAMPLES: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_FORMAT, _AA_SAMPLES, esrFormat));

        dprintf(" PROP%d:    MMU RESP: 0x%x. PTE_KIND: 0x%x, APERTURE: 0x%x\n", gpcIdx,
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _MMU_RESP, esrState),
                    pGr[indexGpu].grGetPteKindFromPropHwwEsr(esrState),
                    DRF_VAL(_PGPC, _PRI_PROP_HWW_ESR_STATE, _APERTURE, esrState));
    }

    if (hwwEsr & DRF_DEF(_PGPC, _PRI_PROP_HWW_ESR, _KIND_SYSMEM_3D_CT_VIOL, _PENDING))
    {
        dprintf("           _GPC%d_PROP_HWW_ESR_KIND_SYSMEM_3D_CT_VIOL\n", gpcIdx);
        dprintf(" Graphics PROP Exception Type:    PROP 3D-C SYSMEM KIND Violation on GPC %d\n", gpcIdx);

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

void grDumpTpcTexExceptionState_GK104(LwU32 gpcCounter, LwU32 tpcCounter)
{
    LwU32 hwwEsr, esrReq, esrAddr, esrMMU, fetchMask;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcCounter,
            GR_UNIT_TYPE_TPC, tpcCounter),
        return);

    dprintf("_TPC%d_TPCCS_TPC_EXCEPTION_TEX_PENDING\n", tpcCounter);
    hwwEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_HWW_ESR);
    esrReq = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_HWW_ESR_REQ);
    esrAddr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_HWW_ESR_ADDR);
    esrMMU = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_TEX_M_HWW_ESR_MMU);

    fetchMask = 0;
    if (esrAddr)
    {
        for ( ; fetchMask < DRF_SIZE(LW_PTPC_PRI_TEX_M_HWW_ESR_REQ_FETCH_MASK); fetchMask++)
        {
            if (DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _FETCH_MASK, esrReq) & (0x1 << fetchMask))
                break;
        }
    }

    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR:0x%x", gpcCounter, tpcCounter, hwwEsr);
    dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_REQ:0x%x\n", gpcCounter, tpcCounter, esrReq);

    dprintf("Graphics TEX Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR 0x%x\n", gpcCounter, tpcCounter, hwwEsr );
    dprintf("Graphics TEX Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_REQ 0x%x\n", gpcCounter, tpcCounter, esrReq );

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TEX_M_HWW_ESR, _FORMAT, _PENDING))
    {
        dprintf(" LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_FORMAT\n", gpcCounter, tpcCounter);
        dprintf(" Graphics TEX Exception Type:    TEX FORMAT Exception on GPC %d, TPC %d\n", gpcCounter, tpcCounter);

        dprintf(" TEX%d:    PADDR LO: 0x%x\n", tpcCounter, (esrAddr << 8) | (fetchMask * 0x20));
        dprintf(" TEX%d:    PADDR HI: 0x%x\n", tpcCounter, (esrAddr & 0xFF) >> 24);

        dprintf(" LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_MMU:0x%x\n", gpcCounter, tpcCounter, esrMMU);

        dprintf(" TEX%d:    Texture header component size (texel format) mismatch with KIND\n", tpcCounter);
        dprintf(" TEX%d:    COMPONENT_SIZE: 0x%x, KIND: 0x%x\n", tpcCounter,
                 DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _COMPONENT_SIZES, esrReq),
                 DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _KIND, esrMMU));
        dprintf(" TEX%d:    Texture header OffsetLower: 0x%x\n", tpcCounter,
                 DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _BASE_DIV_256, esrMMU) << 8);
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TEX_M_HWW_ESR, _LAYOUT, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_FORMAT_LAYOUT\n", gpcCounter, tpcCounter);
        dprintf("Graphics TEX Exception Type:    TEX LAYOUT Exception on GPC %d, TPC %d\n", gpcCounter, tpcCounter);

        dprintf("TEX%d:    PADDR LO: 0x%x\n", tpcCounter, (esrAddr << 8) | (fetchMask * 0x20));
        dprintf("TEX%d:    PADDR HI: 0x%x\n", tpcCounter, (esrAddr & 0xFF) >> 24);

        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_MMU:0x%x\n",gpcCounter, tpcCounter, esrMMU);

        dprintf("TEX%d:    Texture header memory layout mismatch with KIND\n", tpcCounter);
        dprintf("TEX%d:    Texture header memory layout: 0x%x, KIND: 0x%x\n", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _MEMORY_LAYOUT, esrReq),
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _KIND, esrMMU));
        dprintf("TEX%d:    Texture header OffsetLower: 0x%x\n", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _BASE_DIV_256, esrMMU) << 8);
   }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TEX_M_HWW_ESR, _APERTURE, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_FORMAT_APERTURE\n", gpcCounter, tpcCounter);
        dprintf("Graphics TEX Exception Type:    TEX APERTURE Exception on GPC %d, TPC %d\n", gpcCounter, tpcCounter);

        dprintf("TEX%d:    PADDR LO: 0x%x\n", tpcCounter, (esrAddr << 8) | (fetchMask * 0x20));
        dprintf("TEX%d:    PADDR HI: 0x%x\n", tpcCounter, (esrAddr & 0xFF) >> 24);

        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_MMU:0x%x\n", gpcCounter, tpcCounter, esrMMU);

        dprintf("TEX%d:    Illegal sysmem aperture with specified KIND, or PEER aperture mismatch with SUBKIND\n", tpcCounter);
        dprintf("lw TEX%d:    APERTURE: 0x%x, KIND: 0x%x, SUBKIND: 0x%x\n", tpcCounter,
                    DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _APERTURE, esrMMU),
                    DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _KIND, esrMMU),
                    DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _SUBKIND, esrReq));

        dprintf("lw TEX%d:    Texture header OffsetLower: 0x%x\n", tpcCounter,
                    DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _BASE_DIV_256, esrMMU) << 8);
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_TEX_M_HWW_ESR, _NACK, _PENDING))
    {
        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_FORMAT_NACK\n", gpcCounter, tpcCounter);
        dprintf("Graphics TEX Exception Type:    TEX NACK / Page Fault Exception on GPC %d, TPC %d\n", gpcCounter, tpcCounter);

        dprintf("TEX%d:    VADDR: 0x%02x%08x\n", tpcCounter,
                (esrAddr & 0xFF) >> 24, (esrAddr << 8) | (fetchMask * 0x20));

        dprintf("LW_PGRAPH_PRI_GPC%d_TPC%d_TEX_M_HWW_ESR_NACK\n", gpcCounter, tpcCounter);

        // mmu data
        dprintf("TEX%d:    MMU APERTURE: 0x%x,  ", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _APERTURE, esrMMU));

        dprintf("MMU KIND: 0x%x, ",
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _KIND, esrMMU));

        dprintf("MMU Texture Header OffsetLower: 0x%x\n",
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_MMU, _BASE_DIV_256, esrMMU) << 8);

        dprintf("TEX%d:    REQ_FETCH_MASK:0x%x, ", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _FETCH_MASK, esrReq));

        pGr[indexGpu].grDumpTexMHwwEsrReq(tpcCounter,esrReq);

        dprintf("   REQ_SUBKIND:0x%x, REQ_REPLACEMENT:0x%x\n",
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _SUBKIND, esrReq),
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _REPLACEMENT, esrReq));

        dprintf("TEX%d:    REG_MEMORY_LAYOUT:0x%x, REQ_COMPONENT_SIZES:0x%x\n", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _MEMORY_LAYOUT, esrReq),
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _COMPONENT_SIZES, esrReq));

        dprintf("TEX%d:    REQ_SIZE_SWAP:0x%x, REQ_MASK_ROTATE_RIGHT:0x%x\n", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _SIZE_SWAP, esrReq),
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _MASK_ROTATE_RIGHT, esrReq));
    }

}

void grDumpTexMHwwEsrReq_GF100(LwU32 tpcCounter,LwU32 esrReq)
{
        dprintf("TEX%d:    REQ_D_SLOT_ID:0x%x, REG_SUPERSET_ID:0x%x, ", tpcCounter,
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _D_SLOT_ID, esrReq),
                DRF_VAL(_PTPC, _PRI_TEX_M_HWW_ESR_REQ, _SUPERSET_ID, esrReq));
}

typedef struct 
{
    LwU32 AddrMthd;
    LwU32 AddrSubch;
} TrapAddrInfo;

static void _dumpTrappedAddress_GF100(TrapAddrInfo *pTrapInfo)
{
    LwU32 data32 = 0;
    BOOL bFill = !(pTrapInfo == NULL);
    LwU32 value = 0;
    data32 = GPU_REG_RD32(LW_PGRAPH_TRAPPED_ADDR);

    if (data32 & DRF_DEF(_PGRAPH, _TRAPPED_ADDR, _STATUS, _ILWALID))
    {
        dprintf("LW_PGRAPH_TRAPPED_ADDR_STATUS_ILWALID: The trapped method "
                "does not indicate anything about the source of an interrupt\n");
    }
    else
    {
        dprintf("LW_PGRAPH_TRAPPED_ADDR_STATUS_VALID\n");
    }

    value = DRF_VAL(_PGRAPH, _TRAPPED_ADDR, _MTHD, data32) << 2;
    if (bFill)
    {
        pTrapInfo->AddrMthd = value;
    }
    dprintf("        _TRAPPED_ADDR_MTHD:       0x%08x\n", value);

    value = DRF_VAL(_PGRAPH, _TRAPPED_ADDR, _SUBCH, data32);
    if (bFill)
    {
        pTrapInfo->AddrSubch = value;
    }
    dprintf("        _TRAPPED_ADDR_SUBCH:      0x%03x\n", value);

    if (data32 & DRF_DEF(_PGRAPH, _TRAPPED_ADDR, _PRIV, _ENABLED))
        dprintf("    _TRAPPED_ADDR_PRIV_ENABLED\n");
    else
        dprintf("    _TRAPPED_ADDR_PRIV_DISABLED\n");

    dprintf("        _TRAPPED_DATA_LOW_VALUE:  0x%08x\n",
        GPU_REG_RD_DRF(_PGRAPH, _TRAPPED_DATA_LOW, _VALUE));

    //print LW_PGRAPH_TRAPPED_DATA_HIGH if it is valid
    if (data32 & DRF_DEF(_PGRAPH, _TRAPPED_ADDR, _DATAHIGH, _VALID))
    {
        dprintf("    _TRAPPED_ADDR_DATAHIGH_VALID\n");
        dprintf("    _TRAPPED_DATA_HIGH_VALUE:  0x%08x\n",
            GPU_REG_RD_DRF(_PGRAPH, _TRAPPED_DATA_HIGH, _VALUE));
    }
    else
    {
        dprintf("    _TRAPPED_ADDR_DATAHIGH_ILWALID\n");
    }

     //print LW_PGRAPH_TRAPPED_DATA_MME if it is valid
    if (data32 & DRF_DEF(_PGRAPH, _TRAPPED_ADDR, _MME_GENERATED, _TRUE))
    {
        dprintf("    _TRAPPED_ADDR_MME_GENERATED_TRUE\n");
        dprintf("    _TRAPPED_DATA_MME_PC:  0x%08x\n",
            GPU_REG_RD_DRF(_PGRAPH, _TRAPPED_DATA_MME, _PC));
    }
    else
    {
        dprintf("    _TRAPPED_ADDR_MME_GENERATED_FALSE\n");
    }
}

void grCheckIllegalClassInterrupts_GK104
(
    LwU32 grIntr,
    LwU32 grIntrEn
)
{
    TrapAddrInfo trapAddrInfo = {0};
    LwU32 temp = 0;

    if (DRF_VAL(_PGRAPH, _INTR_EN, _ILLEGAL_CLASS, grIntrEn) == 0)
        dprintf("          _EN_ILLEGAL_CLASS_DISABLED\n");

    //5. INTR_ILLEGAL_CLASS
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _ILLEGAL_CLASS, _PENDING))
    {
        dprintf("       _ILLEGAL_CLASS_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_ILLEGAL_CLASS_PENDING\n");

        //get subch info from PGRAPH:TRAPPED
        _dumpTrappedAddress_GF100(&trapAddrInfo);

        //get class for subch from PGRAPH:PRI_FE_OBJECT_TABLE
        temp = GPU_REG_RD32(LW_PGRAPH_PRI_FE_OBJECT_TABLE(trapAddrInfo.AddrSubch));
        dprintf("LW_PGRAPH_PRI_FE_OBJECT_TABLE_LWCLASS:  0x%04x\n",
            DRF_VAL(_PGRAPH, _PRI_FE_OBJECT_TABLE, _LWCLASS, temp));
    }
}

/*!
 *  grDumpGpccsExceptionState_GK104
 */
void grDumpGpccsExceptionState_GK104(LwU32 gpcCounter, LwU32 tpcCounter) 
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

    // is it TEX exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _TEX, _PENDING))
    {
        pGr[indexGpu].grDumpTpcTexExceptionState(gpcCounter, tpcCounter);
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

    // is it L1 exception?
    if (tpccsException & DRF_DEF(_PTPC, _PRI_TPCCS_TPC_EXCEPTION, _L1C, _PENDING))
    {
        pGr[indexGpu].grDumpTPCL1CExceptionState(gpcCounter, tpcCounter);
    }
}

/*!
 *  grDumpPgraphExceptionsState_GK104
 */
void grDumpPgraphExceptionsState_GK104(LwU32 grIdx)
{
    LwU32   data32 = 0;
    LwU32   regEn = GPU_REG_RD32(LW_PGRAPH_EXCEPTION_EN);
    LwU32   regExc = GPU_REG_RD32(LW_PGRAPH_EXCEPTION);
    LwU32   regExcOne = 0;
    LwU32   fbpsCount = 0, gpcCount = 0, maxGpcCount = 0, maxFbpsCount = 0, maxTpcPerGpcCount = 0;
    LwU32   gpcCounter = 0, gpccsException = 0, numTPC = 0, tpcCounter = 0;
    LwU32   tpcShift = 0, tpcMask = 0;
    LW_STATUS status;
    GR_IO_APERTURE *pGpcAperture;

    
    regExc &= regEn;

    dprintf("====================\n");
    dprintf("Checking LW_PGRAPH_EXCEPTION_EN\n");
    dprintf("====================\n");

    //print disabled exceptions
    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_FE, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_FE_DISABLED)
        dprintf("         _EXCEPTION_EN_FE_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_MEMFMT, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_MEMFMT_DISABLED)
        dprintf("         _EXCEPTION_EN_MEMFMT_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_PD, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_PD_DISABLED)
        dprintf("         _EXCEPTION_EN_PD_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_SCC, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_SCC_DISABLED)
        dprintf("         _EXCEPTION_EN_SSC_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_DS, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_DS_DISABLED)
        dprintf("         _EXCEPTION_EN_DS_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_SSYNC, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_SSYNC_DISABLED)
        dprintf("         _EXCEPTION_EN_SSYNC_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_MME, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_MME_DISABLED)
        dprintf("         _EXCEPTION_EN_MME_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION_EN,_GPC, regEn) ==
                LW_PGRAPH_EXCEPTION_EN_GPC_DISABLED)
        dprintf("         _EXCEPTION_EN_GPC_DISABLED\n");

    pGr[indexGpu].grDumpPgraphBeExceptionEnState(regEn); 

    pGr[indexGpu].grPrintMmeFe1Disabled(regEn);

    //print exceptions
    //FE
    // In case of an exception :
    // 1. Check PRI_<unit>_HWW_ESR
    // 2. Check GPC bit.
    // 3. Check BE bit
    dprintf("====================\n");
    dprintf("Checking LW_PGRAPH_EXCEPTION_FE_PENDING for pending exceptions\n");
    dprintf("====================\n");

    if (DRF_VAL(_PGRAPH, _EXCEPTION,_FE, regExc) ==
                LW_PGRAPH_EXCEPTION_FE_PENDING)
    {
        dprintf("        _EXCEPTION_FE_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_FE_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_FE_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_FE_HWW_ESR, _EN,_ENABLE))
            dprintf("LW_PGRAPH_PRI_FE_HWW_ESR_EN_ENABLE\n");
        else
            dprintf("LW_PGRAPH_PRI_FE_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_FE_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_FE_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_FE_HWW_ESR_RESET_ACTIVE\n");
        }

        if (data32 & DRF_DEF(_PGRAPH, _PRI_FE_HWW_ESR, _INJECTED_BUNDLE_ERROR,_PENDING))
            dprintf("LW_PGRAPH_PRI_FE_HWW_ESR_INJECTED_BUNDLE_ERROR_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_FE_HWW_ESR_INJECTED_BUNDLE_ERROR_NOT_PENDING\n");

        //check  state
        pGr[indexGpu].grCheckCtxswStatus(TRUE, grIdx);

    }

    //MEMFMT
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_MEMFMT, regExc) == LW_PGRAPH_EXCEPTION_MEMFMT_PENDING)
    {
        dprintf("        _EXCEPTION_MEMFMT_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_MEMFMT_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MEMFMT_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MEMFMT_HWW_ESR, _EN,_ENABLE))
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_EN_ENABLE\n");
        else
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MEMFMT_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_MEMFMT_HWW_ESR_RESET_ACTIVE\n");
        }

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MEMFMT_HWW_ESR, _EXTRA_INLINE_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_EXTRA_INLINE_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_EXTRA_INLINE_DATA_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MEMFMT_HWW_ESR, _MISSING_INLINE_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_MISSING_INLINE_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MEMFMT_HWW_ESR_MISSING_INLINE_DATA_NOT_PENDING\n");
    }

    //PD
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_PD, regExc) == LW_PGRAPH_EXCEPTION_PD_PENDING)
    {
        dprintf("        _EXCEPTION_PD_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_PD_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_PD_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_PD_HWW_ESR, _EN,_ENABLE))
            dprintf("LW_PGRAPH_PRI_PD_HWW_ESR_EN_ENABLE\n");
        else
            dprintf("LW_PGRAPH_PRI_PD_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_PD_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_PD_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_PD_HWW_ESR_RESET_ACTIVE\n");
        }
    }

    // SCC
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_SCC, regExc) == LW_PGRAPH_EXCEPTION_SCC_PENDING)
    {
        dprintf("LW_PGRAPH_EXCEPTION_SCC_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_SCC_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_SCC_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_SCC_HWW_ESR, _EN,_ENABLE))
           dprintf("LW_PGRAPH_PRI_SCC_HWW_ESR_EN_ENABLE\n");
        else
           dprintf("LW_PGRAPH_PRI_SCC_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_SCC_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_SCC_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_SCC_HWW_ESR_RESET_ACTIVE\n");
        }

        if (data32 & DRF_DEF(_PGRAPH, _PRI_SCC_HWW_ESR, _LDCONST_OOB, _PENDING))
        {
            dprintf("LW_PGRAPH_PRI_SCC_HWW_ESR_LDCONST_OOB_PENDING\n");
            addUnitErr("\t LW_PGRAPH_PRI_SCC_HWW_ESR_LDCONST_OOB_PENDING\n");

            dprintf(" LW_PGRAPH_PRI_SCC_HWW_ESR_CONSTBUFF info .. \n");

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_SCC_HWW_ESR_CONSTBUFF_BASE);
            dprintf("            _CONSTBUFF_BASE_ADDR_DIV_256B: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _SCC_HWW_ESR_CONSTBUFF_BASE,_ADDR_DIV_256B, data32));

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_SCC_HWW_ESR_CONSTBUFF);
            dprintf("            _CONSTBUFF_CONST_DW_OFFSET: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _SCC_HWW_ESR_CONSTBUFF,_CONST_DW_OFFSET, data32));
            dprintf("            _CONSTBUFF_SIZE_IN_DW: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _SCC_HWW_ESR_CONSTBUFF,_SIZE_IN_DW, data32));
        }
    }

    // DS
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_DS, regExc) == LW_PGRAPH_EXCEPTION_DS_PENDING)
    {
        dprintf("        _EXCEPTION_DS_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_DS_PENDING\n");

        pGr[indexGpu].grDumpDsExceptionState();
    }

    //SSYNC
    //PRI_HWW_ESR hardwired to zero lwrrently.

    //MME
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_MME, regExc) == LW_PGRAPH_EXCEPTION_MME_PENDING)
    {
        dprintf("        _EXCEPTION_MME_PENDING\n");
        addUnitErr("\t LW_PGRAPH_EXCEPTION_MME_PENDING\n");

        data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_HWW_ESR);

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _MISSING_MACRO_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_MISSING_MACRO_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_MISSING_MACRO_DATA_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _EXTRA_MACRO_DATA,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_EXTRA_MACRO_DATA_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_EXTRA_MACRO_DATA_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _ILLEGAL_OPCODE,_PENDING))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_ILLEGAL_OPCODE_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_ILLEGAL_OPCODE_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _BRANCH_IN_DELAY_SLOT, _PENDING))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_BRANCH_IN_DELAY_SLOT_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_BRANCH_IN_DELAY_SLOT_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _MAX_INSTR_LIMIT, _PENDING))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_MAX_INSTR_LIMIT_PENDING\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_MAX_INSTR_LIMIT_NOT_PENDING\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _STOP_ON_TRAP,_DISABLED))
        {
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_STOP_ON_TRAP_DISABLED\n");
            dprintf("Now checking _PRI_MME_ESR_INFO/INFO2\n");

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_HWW_ESR_INFO);
            pGr[indexGpu].grDumpMmeHwwEsr(data32);

            data32 = GPU_REG_RD32(LW_PGRAPH_PRI_MME_HWW_ESR_INFO2);
            dprintf("            _PRI_MME_HWW_ESR_INFO2_IR: 0x%x\n",
                     DRF_VAL(_PGRAPH_PRI, _MME_HWW_ESR_INFO2, _IR, data32));
        }
        else
        {
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_STOP_ON_TRAP_ENABLED\n");
        }

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _EN,_ENABLE))
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_EN_ENABLE\n");
        else
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_EN_DISABLE\n");

        if (data32 & DRF_DEF(_PGRAPH, _PRI_MME_HWW_ESR, _RESET, _ACTIVE))
        {
            dprintf("LW_PGRAPH_PRI_MME_HWW_ESR_RESET_ACTIVE\n");
            addUnitErr("\t LW_PGRAPH_PRI_MME_HWW_ESR_RESET_ACTIVE\n");
        }

    }//end MME EXCEPTION

    //GPC
    if (DRF_VAL(_PGRAPH, _EXCEPTION,_GPC, regExc) == LW_PGRAPH_EXCEPTION_GPC_PENDING)
    {
        pGr[indexGpu].grGetBusInfo(&gpcCount, &maxGpcCount, &fbpsCount, &maxFbpsCount, &maxTpcPerGpcCount, grIdx );

        regExcOne = GPU_REG_RD32(LW_PGRAPH_EXCEPTION1);
        dprintf("   LW_PGRAPH_EXCEPTION1:    0x%x\n", regExcOne);

        for (gpcCounter = 0; gpcCounter < gpcCount; gpcCounter++)
        {
            LW_ASSERT_OK_OR_ELSE(status,
                GR_GET_APERTURE(&grApertures[indexGpu], &pGpcAperture, GR_UNIT_TYPE_GPC, gpcCounter),
                return);

            if (regExcOne & BIT(gpcCounter))
            {
                // GPC (i) has an exception
                gpccsException = REG_RD32(&pGpcAperture->aperture, LW_PGPC_PRI_GPCCS_GPC_EXCEPTION);
                dprintf("LW_PGRAPH_PRI_GPC%d_GPCCS_GPC_EXCEPTION:    0x%x\n", gpcCounter, gpccsException);

                // is it PROP exception?
                if (gpccsException & DRF_DEF(_PGPC, _PRI_GPCCS_GPC_EXCEPTION, _PROP, _PENDING))
                {
                    pGr[indexGpu].grDumpPropExceptionState(gpcCounter);
                }

                // is it ZLWLL exception?
                if (gpccsException & DRF_DEF(_PGPC, _PRI_GPCCS_GPC_EXCEPTION, _ZLWLL, _PENDING))
                {
                    pGr[indexGpu].grDumpZlwllExceptionState(gpcCounter);
                }

                // is it GCC exception
                if  (gpccsException & DRF_DEF(_PGPC, _PRI_GPCCS_GPC_EXCEPTION, _GCC, _PENDING))
                {
                    pGr[indexGpu].grDumpGccExceptionState(gpcCounter);
                }

                // is it SETUP exception?
                if (gpccsException & DRF_DEF(_PGPC, _PRI_GPCCS_GPC_EXCEPTION, _SETUP, _PENDING))
                {
                    pGr[indexGpu].grDumpSetupExceptionState(gpcCounter);
                }

                // is it a TPC exception?
                numTPC = pGr[indexGpu].grGetNumTpcForGpc( gpcCounter, grIdx );
                for (tpcCounter = 0; tpcCounter < numTPC; tpcCounter++)
                {
                    // create mask for this TPC
                    tpcMask = pGr[indexGpu].grTpcExceptionMask(tpcCounter);

                    if (gpccsException & tpcMask)
                    {
                        pGr[indexGpu].grDumpGpccsExceptionState(gpcCounter, tpcCounter);
                    }
                    
                }

                pGr[indexGpu].grDumpRopExceptionState(gpcCounter, grIdx);
            }
        }
    }
    pGr[indexGpu].grDumpPgraphBeExceptionsState(regExc, fbpsCount);
}

/*!
 *  grTpcExceptionMask_GK104
 *  Return TPC exception mask for identifiying the TPC 
 *  which has the GPCCS exception. 
 */
LwU32 grTpcExceptionMask_GK104(LwU32 tpcIdx)
{
    // create mask for tpcIdx TPC
    return (1 << (DRF_SHIFT(LW_PGPC_PRI_GPCCS_GPC_EXCEPTION_TPC) + tpcIdx));
}

/*!
 *  grCheckGrInterrupts_GK104
 */
LW_STATUS grCheckGrInterrupts_GK104( LwU32 grIdx )
{
    LwU32           grIntr = 0;
    LwU32           grIntrEn = 0;
    LwU32           grIntrRoute = 0;
    TrapAddrInfo    trapAddrInfo = {0};
    LwU32           temp = 0;

    //read interrupts and mask w/ enable
    grIntr = GPU_REG_RD32(LW_PGRAPH_INTR);
    grIntrEn = GPU_REG_RD32(LW_PGRAPH_INTR_EN);

    grIntrRoute = GPU_REG_RD32(LW_PGRAPH_INTR_ROUTE);

    dprintf("Checking lwrrently disabled intrs at LW_PGRAPH_INTR_EN...\n");

    //print disabled interrupts
    if (DRF_VAL(_PGRAPH, _INTR_EN, _NOTIFY, grIntrEn) == 0)
        dprintf("          _EN_NOTIFY_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _SEMAPHORE, grIntrEn) == 0)
        dprintf("          _EN_SEMAPHORE_DISABLED\n");

    pGr[indexGpu].grDumpIntrEnSemaphoreTimeout(grIntrEn);

    if (DRF_VAL(_PGRAPH, _INTR_EN, _ILLEGAL_METHOD, grIntrEn) == 0)
    {
        dprintf("          _EN_ILLEGAL_METHOD_DISABLED\n");
        if (DRF_VAL(_PGRAPH, _INTR_ROUTE, _ILLEGAL_METHOD, grIntrRoute) ==
            LW_PGRAPH_INTR_ROUTE_ILLEGAL_METHOD_FECS)
        {
            dprintf("            _ILLEGAL_METHOD routed to FECS\n");
        }
        else
        {
            dprintf("            _ILLEGAL_METHOD routed to HOST\n");
        }
    }

    if (DRF_VAL(_PGRAPH, _INTR_EN, _ILLEGAL_NOTIFY, grIntrEn) == 0)
        dprintf("          _EN_ILLEGAL_NOTIFY_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _DEBUG_METHOD, grIntrEn) == 0)
        dprintf("          _EN_DEBUG_METHOD_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _FIRMWARE_METHOD, grIntrEn) == 0)
    {
        dprintf("          _EN_FIRMWARE_METHOD_DISABLED\n");
        if (DRF_VAL(_PGRAPH, _INTR_ROUTE, _FIRMWARE_METHOD, grIntrRoute) ==
            LW_PGRAPH_INTR_ROUTE_FIRMWARE_METHOD_FECS)
        {
            dprintf("             _FIRMWARE_METHOD routed to FECS\n");
        }
        else
        {
            dprintf("             _FIRMWARE_METHOD routed to HOST\n");
        }
    }

    if (DRF_VAL(_PGRAPH, _INTR_EN, _BUFFER_NOTIFY, grIntrEn) == 0)
        dprintf("          _EN_BUFFER_NOTIFY_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _FECS_ERROR, grIntrEn) == 0)
        dprintf("          _EN_CLASS_ERROR_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _CLASS_ERROR, grIntrEn) == 0)
        dprintf("          _EN_CLASS_ERROR_DISABLED\n");

    if (DRF_VAL(_PGRAPH, _INTR_EN, _EXCEPTION, grIntrEn) == 0)
        dprintf("          _EN_EXCEPTION_DISABLED\n");

    if (grIntr == 0)
    {
        dprintf("No gr interrupts pending lwrrently\n");
        return LW_OK;
    }

    dprintf("Gr Interrupts pending:    0x%08x\n", grIntr);
    addUnitErr("\t Gr Interrupts pending:    0x%08x\n", grIntr);

    //1. INTR_NOTIFY
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _NOTIFY, _PENDING))
    {
        dprintf("        _NOTIFY_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_NOTIFY_PENDING\n");
    }
    //2. INTR_SEMAPHORE
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _SEMAPHORE, _PENDING))
    {
        dprintf("       _SEMAPHORE_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_SEMAPHORE_PENDING\n");
        pGr[indexGpu].grDumpPriFeSemaphoreState();
    }

    //3. INTR_SEMAPHORE_TIMEOUT
    pGr[indexGpu].grDumpIntrSemaphoreTimeout(grIntr);

    //4. INTR_ILLEGAL_METHOD
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _ILLEGAL_METHOD, _PENDING))
    {
        dprintf("       _ILLEGAL_METHOD_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_ILLEGAL_METHOD_PENDING\n");

        //dump info if INTR_ROUTE is set to host.
        if (DRF_VAL(_PGRAPH, _INTR_ROUTE, _ILLEGAL_METHOD, grIntrRoute) ==
            LW_PGRAPH_INTR_ROUTE_ILLEGAL_METHOD_HOST)
        {
            _dumpTrappedAddress_GF100(&trapAddrInfo);
        }

        if (pGr[indexGpu].grCheckFeMethodStatus() == LW_ERR_GENERIC)
        {
            dprintf("FE is in invalid state.\n");
            addUnitErr("\t FE is in invalid state: method status is invalid\n");
        }
    }

    //5. INTR_ILLEGAL_CLASS
    pGr[indexGpu].grCheckIllegalClassInterrupts(grIntr, grIntrEn);

    //TODO: Add more analysis
    //6. INTR_ILLEGAL_NOTIFY

    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _ILLEGAL_NOTIFY, _PENDING))
    {
        dprintf("       _ILLEGAL_NOTIFY_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_ILLEGAL_NOTIFY_PENDING\n");

        _dumpTrappedAddress_GF100(NULL);
    }

    //7. INTR_FIRMWARE_METHOD
    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _FIRMWARE_METHOD, _PENDING))
    {
        dprintf("       _ILLEGAL_METHOD_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_FIRMWARE_METHOD_PENDING\n");

        _dumpTrappedAddress_GF100(&trapAddrInfo);
    }

    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _BUFFER_NOTIFY, _PENDING))
    {
        dprintf("       _BUFFER_NOTIFY_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_BUFFER_NOTIFY_PENDING\n");
    }

    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _FECS_ERROR, _PENDING))
    {
        dprintf("       _FECS_ERROR_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_CLASS_ERROR_PENDING\n");
    }

    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _CLASS_ERROR, _PENDING))
    {
        dprintf("       _CLASS_ERROR_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_CLASS_ERROR_PENDING\n");

        dprintf("LW_PGRAPH_CLASS_ERROR_CODE:    0x%04x. Check dev_mmu.ref\n",
            GPU_REG_RD_DRF(_PGRAPH, _CLASS_ERROR, _CODE) );
        _dumpTrappedAddress_GF100(NULL);
    }

    if (grIntr & DRF_DEF(_PGRAPH, _INTR, _EXCEPTION, _PENDING))
    {
        dprintf("LW_PGRAPH_INTR_EXCEPTION_PENDING\n");
        addUnitErr("\t LW_PGRAPH_INTR_EXCEPTION_PENDING\n");

        pGr[indexGpu].grDumpPgraphExceptionsState( grIdx );
    }

    pGr[indexGpu].grDumpIntrSingleStep(grIntr);
    return LW_ERR_GENERIC;
}

void grDumpTpcPeExceptionState_GK104(LwU32 gpcCounter, LwU32 tpcCounter)
{
    LwU32 hwwEsr;
    GR_IO_APERTURE *pTpcAperture;
    LW_STATUS status;
    dprintf(" _TPC%d_TPCCS_TPC_EXCEPTION_PE_PENDING\n", tpcCounter);

    LW_ASSERT_OK_OR_ELSE(status,
        GR_GET_APERTURE(&grApertures[indexGpu], &pTpcAperture, GR_UNIT_TYPE_GPC, gpcCounter,
            GR_UNIT_TYPE_TPC, tpcCounter),
        return);

    hwwEsr = REG_RD32(&pTpcAperture->aperture, LW_PTPC_PRI_PE_HWW_ESR);

    dprintf("Graphics PE Exception: LW_PGRAPH_PRI_GPC%d_TPC%d_PE_HWW_ESR 0x%x\n", gpcCounter, tpcCounter, hwwEsr);

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _PIF_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: PIF_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _PIN_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: PIN_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _VAF_ERR , _PENDING))
    {
        dprintf("Graphics PE Exception Type: VAF_ERR \n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _ACACHE_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: ACACHE_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _STRI_ERR, _PENDING))
    {
      dprintf("Graphics PE Exception Type: STRI_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _TG_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: TG_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _GSPILL_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: GSPILL_ERR\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _GPULL_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: GPULL_ER\n");
    }

    if (hwwEsr & DRF_DEF(_PTPC, _PRI_PE_HWW_ESR, _VSC_ERR, _PENDING))
    {
        dprintf("Graphics PE Exception Type: VSC_ERR\n");
    }
}

