/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "pascal/gp100/dev_top.h"
#include "pascal/gp100/dev_ce.h"
#include "pascal/gp100/dev_ce_addendum.h"

#include "gpuanalyze.h"
#include "hwref/lwutil.h"
#include "fifo.h"
#include "deviceinfo.h"
#include "ce.h"
#include "hal.h"
#include "g_ce_hal.h"
#include "chip.h"


//-----------------------------------------------------
// ceIsValid_GP100
//-----------------------------------------------------
BOOL ceIsValid_GP100(LwU32 indexCe)
{
    LwU32 numCes;

    //
    // CE fuses will be deprecated for Pascal+. Hence, return
    // true if indexCe is within the max CEs from ptop
    //
    numCes = GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_CES, _VALUE);

    return (indexCe <= numCes - 1);
}

//-----------------------------------------------------
// ceIsPresent_GP100
//-----------------------------------------------------
BOOL ceIsPresent_GP100(LwU32 indexCe)
{
    LwU32 i;

    //
    // CE fuses will be deprecated for Pascal+. Hence, look
    // into device info to determine if CE is present in ptop
    //
    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        for (i = 0; i < deviceInfo.enginesCount; ++i)
        {
            if ( (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                     == LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE) &&
                 (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INST_ID]
                     == indexCe) )
            {
                return TRUE;
            }
        }
    }

    return FALSE;
}

//----------------------------------------------------------
// ceIsEnabled_GP100
//     Return true if at least one PCE is assigned to the CE
//----------------------------------------------------------
BOOL ceIsEnabled_GP100(LwU32 indexGpu, LwU32 indexCe)
{
    LwU32  pcemap;
    LwU32  i;

    // Get the mask of all enabled PCEs
    pcemap = GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);

    FOR_EACH_INDEX_IN_MASK(32, i, pcemap)
    {
        if (pCe[indexGpu].ceIsPceAssignedToLce(indexGpu, i, indexCe))
        {
            return TRUE;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return FALSE;
}

//-----------------------------------------------------
// ceIsPceAssignedToCe_GP100: Determine if the pce is
//                          assigned to the given lce
//-----------------------------------------------------
BOOL ceIsPceAssignedToLce_GP100
(
    LwU32 indexGpu,
    LwU32 pce,
    LwU32 indexCe
)
{
    LwU32  lwrCe;

    lwrCe  = DRF_IDX_VAL(_CE, _PCE2LCE_CONFIG0, _IDX, pce,
                            GPU_REG_RD32(LW_CE_PCE2LCE_CONFIG0));

    if (indexCe == lwrCe)
    {
        return TRUE;
    }
    else
    {
        // If GRCE is shared
        if ( (indexCe == pCe[indexGpu].ceGetGrce(indexGpu)) &&
                GPU_REG_RD_DRF(_CE, _GRCE_CONFIG, _SHARED) )
        {
            return (lwrCe == GPU_REG_RD_DRF(_CE, _GRCE_CONFIG, _SHARED_LCE));
        }
        return FALSE;
    }
}

//-----------------------------------------------------
// ceGetGrce_GP100: Return GRCE index
//-----------------------------------------------------
LwU32 ceGetGrce_GP100(LwU32 indexGpu)
{
    LwU32 i;
    LwU32 runList = -1;

    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        // Get the runlist of GR engine
        for (i = 0; i < deviceInfo.enginesCount; ++i)
        {
            if (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                    == LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS)
            {
                runList = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST];
                break;
            }
        }

        // Get the copy engine index with the same runlist as GR
        for (i = 0; i < deviceInfo.enginesCount; ++i)
        {
            if ( (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                    == LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE) &&
                 (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST]
                    == runList) )
            {
                return deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INST_ID];
            }
        }
    }

    // No GRCOPY exists. Return max CE count from ptop
    return GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_CES, _VALUE);
}

//-----------------------------------------------------
// ceIsSupported_GP100
//-----------------------------------------------------
BOOL ceIsSupported_GP100(LwU32 indexGpu, LwU32 indexCe)
{
    if (!pCe[indexGpu].ceIsValid(indexCe))
    {
        return FALSE;
    }

    //
    // There will be no floor-sweeping of CEs from Pascal+
    // Hence look into device info to determine if CE is present
	//
    if (!pCe[indexGpu].ceIsPresent(indexCe))
    {
        return FALSE;
    }

    // Return true if at least one PCE is assigned to the CE
    return pCe[indexGpu].ceIsEnabled(indexGpu, indexCe);
}

//----------------------------------------------------
// cePrintPceLceMap_GP100: Print the PCE-LCE Mappings
//----------------------------------------------------
void cePrintPceLceMap_GP100(void)
{
    LwU32  pcemap;
    LwU32  regVal;
    LwU32  grce;
    LwU32  i;

    // Get the mask of all enabled PCEs
    pcemap = GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);

    // Read the config0 register to get the PCE-LCE mappings
    regVal = GPU_REG_RD32(LW_CE_PCE2LCE_CONFIG0);

    FOR_EACH_INDEX_IN_MASK(32, i, pcemap)
    {
        if (DRF_IDX_VAL(_CE, _PCE2LCE_CONFIG0, _IDX, i, regVal)
                == LW_CE_PCE2LCE_CONFIG0_IDX_NONE)
        {
            dprintf("lw: PCE%d NOT assigned to any LCE\n", i);
        }
        else
        {
            dprintf("lw: PCE%d assigned to LCE%d\n", i,
                    DRF_IDX_VAL(_CE, _PCE2LCE_CONFIG0, _IDX, i, regVal));
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    dprintf("lw:\n");

    grce = pCe[indexGpu].ceGetGrce(indexGpu);
    if (!pCe[indexGpu].ceIsValid(grce))
    {
        dprintf("lw: NO GRCOPY Engine\n");
    }
    else
    {
        dprintf("lw: GRCOPY Engine is CE%d\n", grce);
        dprintf("lw:\n");

        regVal = GPU_REG_RD32(LW_CE_GRCE_CONFIG);
        if (DRF_VAL(_CE, _GRCE_CONFIG, _SHARED, regVal))
        {
            dprintf("lw: GRCE shared with LCE%d\n",
                    DRF_VAL(_CE, _GRCE_CONFIG, _SHARED_LCE, regVal));
        }
        else
        {
            dprintf("lw: GRCE is NOT shared\n");
        }
    }
    dprintf("lw:\n");
}

/*!
 *  Read the PCE_PIPESTATUS register (the offset changed between Pascal and
 *  Volta, but otherwise the definitions are the same).
 *  @param[in] pce - physical copy engine index
 *
 *  @return LW_CE_PCE_PIPESTATUS value
 */
LwU32 ceReadPcePipestatus_GP100(LwU32 pce)
{
    return GPU_REG_RD32(LW_CE_PCE_PIPESTATUS(pce));
}

/*!
 *  check Ce engine status
 *  @param[in]      indexGpu
 *  @param[in]      ceIndex
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LwU32 ceCheckCeState_GP100( LwU32 indexGpu, LwU32 eng )
{
    LwU32   status = LW_OK;
    LwU32   data32;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   pcemap;
    char    engName[8];
    LwU32   i;

    // Print the pce-lce mappings
    pCe[indexGpu].cePrintPceLceMap();

    //check interrupts
    regIntr = GPU_REG_RD32( LW_CE_LCE_INTR_STATUS(eng) );
    regIntrEn = GPU_REG_RD32( LW_CE_LCE_INTR_EN(eng) );
    regIntr &= regIntrEn;

    sprintf( engName, "LCE%d", eng);

    if ( DRF_VAL(_CE, _LCE_INTR_STATUS, _BLOCKPIPE, regIntr) )
    {
        dprintf("lw: + LW_CE_%s_INTR_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_CE_%s_INTR_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL(_CE, _LCE_INTR_STATUS, _NONBLOCKPIPE, regIntr) )
    {
        dprintf("lw: + LW_CE_%s_INTR_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_CE_%s_INTR_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }
    dprintf("lw: \n");

    // Get the mask of all enabled PCEs
    pcemap = GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);

    // Get the pipe status
    FOR_EACH_INDEX_IN_MASK(32, i, pcemap)
    {
        if (pCe[indexGpu].ceIsPceAssignedToLce(indexGpu, i, eng))
        {
            sprintf( engName, "PCE%d", i);
            data32 = pCe[indexGpu].ceReadPcePipestatus(i);

            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _CTL, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_CTL_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_CTL_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _GSTRIP, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _RALIGN, data32) )
            {
                dprintf("lw: + LW_%s_PIPESTATUS_RALIGN_NONIDLE\n", engName);
                addUnitErr("\t LW_%s_PIPESTATUS_RALIGN_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _SWIZ, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_SWIZ_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_SWIZ_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _WALIGN, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_WALIGN_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_WALIGN_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _GPAD, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_GPAD_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_GPAD_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _RDAT, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_RDAT_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_RDAT_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _RDACK, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_RDACK_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_RDACK_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _WRACK, data32) )
            {
                dprintf("lw: LW_CE_%s_PIPESTATUS_WRACK_NONIDLE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_WRACK_NONIDLE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _RCMD_STALL, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
                status = LW_ERR_GENERIC;
            }
            if ( DRF_VAL(_CE, _PCE_PIPESTATUS, _WCMD_STALL, data32) )
            {
                dprintf("lw: + LW_CE_%s_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
                addUnitErr("\t LW_CE_%s_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
                status = LW_ERR_GENERIC;
            }
            dprintf("lw: \n");
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

//-----------------------------------------------------
// ceDumpPriv_GP100 - Dumps CE priv reg space
//-----------------------------------------------------
LwU32 ceDumpPriv_GP100( LwU32 indexGpu, LwU32 indexCe )
{
    LwU32 u;
    LwU32 pcemap;
    char m_tag[256];

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u CE%d priv registers -- \n", indexGpu, indexCe);
    dprintf("lw:\n");

    // Global Registers
    sprintf(m_tag,"LW_CE_PMM");
    cePrintPriv(40,m_tag, LW_CE_PMM);
    sprintf(m_tag,"LW_CE_ENGCAP");
    cePrintPriv(40,m_tag, LW_CE_ENGCAP);
    sprintf(m_tag,"LW_CE_GRCE_CONFIG");
    cePrintPriv(40,m_tag, LW_CE_GRCE_CONFIG);
    dprintf("lw:\n");

    // INTR/BIND statuses, INTR EN, LAUNCHERR, THROTTLE, ENGCTL
    sprintf(m_tag,"LW_CE_LCE_INTR_EN(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_INTR_EN(indexCe));
    sprintf(m_tag,"LW_CE_LCE_INTR_STATUS(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_INTR_STATUS(indexCe));
    sprintf(m_tag,"LW_CE_LCE_BIND_STATUS(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_BIND_STATUS(indexCe));
    sprintf(m_tag,"LW_CE_LCE_LAUNCHERR(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_LAUNCHERR(indexCe));
    sprintf(m_tag,"LW_CE_LCE_THROTTLE(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_THROTTLE(indexCe));
    sprintf(m_tag,"LW_CE_LCE_ENGCTL(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_ENGCTL(indexCe));
    sprintf(m_tag,"LW_CE_LCE_OPT(%d)",indexCe);
    cePrintPriv(40,m_tag, LW_CE_LCE_OPT(indexCe));
    dprintf("lw:\n");

    // Get the mask of all enabled PCEs
    pcemap = GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);

    // Loop through all PCEs and read registers if the PCE is assigned to CE
    FOR_EACH_INDEX_IN_MASK(32, u, pcemap)
    {
        if(pCe[indexGpu].ceIsPceAssignedToLce(indexGpu, u, indexCe))
        {
            sprintf(m_tag,"LW_CE_PCE_CMD0(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_CMD0(u));
            sprintf(m_tag,"LW_CE_PCE_LINES_TO_COPY(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_LINES_TO_COPY(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM0(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM0(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM0(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM0(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM1(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM1(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM1(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM1(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM2(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM2(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM2(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM2(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM3(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM3(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM3(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM3(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM4(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM4(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM4(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM4(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PARAM5(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PARAM5(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PARAM5(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PARAM5(u));
            sprintf(m_tag,"LW_CE_PCE_SWIZZLE_CONSTANT0(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SWIZZLE_CONSTANT0(u));
            sprintf(m_tag,"LW_CE_PCE_SWIZZLE_CONSTANT1(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SWIZZLE_CONSTANT1(u));
            sprintf(m_tag,"LW_CE_PCE_CMD1(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_CMD1(u));
            sprintf(m_tag,"LW_CE_PCE_SRC_PHYS_MODE(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_SRC_PHYS_MODE(u));
            sprintf(m_tag,"LW_CE_PCE_DST_PHYS_MODE(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_DST_PHYS_MODE(u));
            sprintf(m_tag,"LW_CE_PCE_PIPESTATUS(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_PIPESTATUS(u));
            dprintf("lw:\n");
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LW_OK;
}
