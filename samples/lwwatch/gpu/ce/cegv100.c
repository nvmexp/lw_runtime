/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "volta/gv100/dev_top.h"
#include "volta/gv100/dev_ce.h"

#include "hwref/lwutil.h"
#include "fifo.h"
#include "deviceinfo.h"
#include "ce.h"
#include "g_ce_hal.h"

/**
 * @brief      Read LW_CE_PCE_MAP register as present PCE mask
 */
LwU32 ceGetPresentPceMask_GV100(void)
{
    return GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);
}

/**
 * @brief      Get lceIdx that the PCE is mapped to
 *
 * @return     FALSE if there is no mapping
 *             TRUE and the lceIdx otherwise
 */
BOOL ceGetPceMap_GV100
(
    LwU32   pceIdx,
    LwU32  *pLceIdx
)
{
    if (pCe[indexGpu].ceIsValid(pceIdx))
    {
        LwU32 regVal = GPU_REG_RD32(LW_CE_PCE2LCE_CONFIG(pceIdx));
        if (regVal == LW_CE_PCE2LCE_CONFIG_PCE_ASSIGNED_LCE_NONE)
        {
            return FALSE;
        }
        else
        {
            *pLceIdx = regVal;
            return TRUE;
        }
    }

    return FALSE;
}

/**
 * @brief      Check if the LCE is GRCE
 * @details    The LCE is GRCE when it has the same runlist as GR engine
 */
BOOL ceIsLceGrce_GV100(LwU32 lceIdx)
{
    LwU32 i;
    LwU32 grRunlist;

    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        // Get grRunlist
        for (i = 0; i < deviceInfo.enginesCount; i++)
        {
            if (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                    == LW_PTOP_DEVICE_INFO_TYPE_ENUM_GRAPHICS)
            {
                grRunlist = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST];
                break;
            }
        }

        // No grRunlist available, return false
        if (i == deviceInfo.enginesCount)
        {
            return FALSE;
        }

        // Check if the LCE has the same runlist as GR engine
        for (i = 0; i < deviceInfo.enginesCount; ++i)
        {
            if ( (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                    == LW_PTOP_DEVICE_INFO_TYPE_ENUM_LCE) &&
                 (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST]
                    == grRunlist) )
            {
                if (lceIdx ==
                    deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INST_ID])
                {
                    return TRUE;
                }
            }
        }
    }

    return FALSE;
}

/**
 * @brief      Print all pce lce mapping info, and GRCE info
 */
void cePrintPceLceMap_GV100(void)
{
    LwU32 pceMap = pCe[indexGpu].ceGetPresentPceMask();
    LwU32 i = 0;
    LwU32 grceCount = 0;
    LwU32 ceLceStatusSize = pCe[indexGpu].ceGetCeLceStatusSize();

    // print PCE-LCE mapping info
    FOR_EACH_INDEX_IN_MASK(32, i, pceMap)
    {
        LwU32 lceIdx = 0;
        if (pCe[indexGpu].ceGetPceMap(i, &lceIdx))
        {
            dprintf("lw: PCE%d assigned to LCE%d\n", i, lceIdx);
        }
        else
        {
            dprintf("lw: PCE%d NOT assigned to any LCE\n", i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
    dprintf("lw:\n");

    // print GRCE info
    for (i = 0; i < ceLceStatusSize; i++)
    {
        if (pCe[indexGpu].ceIsLceGrce(i))
        {
            LwU32 grceConfig = GPU_REG_RD32(LW_CE_GRCE_CONFIG(grceCount));

            dprintf("lw: LCE%d is GRCE, ", i);
            if (DRF_VAL(_CE, _GRCE_CONFIG, _SHARED, grceConfig))
            {
                dprintf("shared with LCE%d\n",
                    DRF_VAL(_CE, _GRCE_CONFIG, _SHARED_LCE, grceConfig));
            }
            else
            {
                dprintf("not shared\n");
            }

            grceCount++;
            if (grceCount == LW_CE_GRCE_CONFIG__SIZE_1)
                break;
        }
    }

    dprintf("lw:\n");
}

//-----------------------------------------------------
// ceIsPceAssignedToCe_GV100: Determine if the pce is
//                          assigned to the given lce
//-----------------------------------------------------
BOOL ceIsPceAssignedToLce_GV100
(
    LwU32 indexGpu,
    LwU32 pce,
    LwU32 indexCe
)
{
    LwU32 lwrCe;
    LwU32 pceToLceConfigSize = pCe[indexGpu].ceGetPceToLceConfigSize();
    
    if (pce >= pceToLceConfigSize)
    {
        return FALSE;
    }
    lwrCe = GPU_REG_IDX_RD_DRF(_CE, _PCE2LCE_CONFIG, pce, _PCE_ASSIGNED_LCE);

    if (indexCe == lwrCe)
    {
        return TRUE;
    }
    else
    {
        // If GRCE is shared
        if (indexCe == pCe[indexGpu].ceGetGrce(indexGpu) &&
            // TODO: GV100 has two GRCE engines, revisit
            lwrCe == GPU_REG_IDX_RD_DRF(_CE, _GRCE_CONFIG, 0, _SHARED_LCE))
        {
            return TRUE;
        }
        return FALSE;
    }
}


//-----------------------------------------------------
// ceDumpPriv_GV100 - Dumps CE priv reg space
//-----------------------------------------------------
LwU32 ceDumpPriv_GV100( LwU32 indexGpu, LwU32 indexCe )
{
    LwU32 u;
    LwU32 pcemap;
    char m_tag[256];

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u CE%d priv registers -- \n", indexGpu, indexCe);
    dprintf("lw:\n");

    // Global Registers
    sprintf(m_tag,"LW_CE_ENGCAP");
    cePrintPriv(40,m_tag, LW_CE_ENGCAP);
    sprintf(m_tag,"LW_CE_GRCE_CONFIG(0)");
    cePrintPriv(40,m_tag, LW_CE_GRCE_CONFIG(0));
    sprintf(m_tag,"LW_CE_GRCE_CONFIG(1)");
    cePrintPriv(40,m_tag, LW_CE_GRCE_CONFIG(1));
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
            sprintf(m_tag,"LW_CE_PCE_GLOBAL_COUNTER_LOWER(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_GLOBAL_COUNTER_LOWER(u));
            sprintf(m_tag,"LW_CE_PCE_GLOBAL_COUNTER_UPPER(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_GLOBAL_COUNTER_UPPER(u));
            sprintf(m_tag,"LW_CE_PCE_PAGEOUT_START_PALOWER(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_PAGEOUT_START_PALOWER(u));
            sprintf(m_tag,"LW_CE_PCE_PAGEOUT_START_PAUPPER(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_PAGEOUT_START_PAUPPER(u));
            sprintf(m_tag,"LW_CE_PCE_PMM(%d)",u);
            cePrintPriv(40,m_tag, LW_CE_PCE_PMM(u));
            dprintf("lw:\n");
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LW_OK;
}

/*!
 *  Read the PCE_PIPESTATUS register (the offset changed between Pascal and
 *  Volta, but otherwise the definitions are the same).
 *  @param[in] pce - physical copy engine index
 *
 *  @return LW_CE_PCE_PIPESTATUS value
 */
LwU32 ceReadPcePipestatus_GV100(LwU32 pce)
{
    return GPU_REG_RD32(LW_CE_PCE_PIPESTATUS(pce));
}

/*!
 *  Get the LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 *
 *  @return LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 */
LwU32 ceGetPceToLceConfigSize_GV100(void)
{
    return LW_CE_PCE2LCE_CONFIG__SIZE_1;
}

/*!
 *  Get the LW_CE_LCE_STATUS__SIZE_1 value
 *
 *  @return LW_CE_LCE_STATUS__SIZE_1 value
 */
LwU32 ceGetCeLceStatusSize_GV100(void)
{
    return LW_CE_LCE_STATUS__SIZE_1;
}
