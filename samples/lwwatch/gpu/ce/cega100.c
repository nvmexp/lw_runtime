/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "ampere/ga100/dev_top.h"
#include "ampere/ga100/dev_ce.h"

#include "hwref/lwutil.h"
#include "fifo.h"
#include "deviceinfo.h"
#include "g_ce_hal.h"
#include "ce.h"

/**
 * @brief      Read LW_CE_PCE_MAP register as present PCE mask
 */
LwU32 ceGetPresentPceMask_GA100(void)
{
    return GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);
}

/**
 * @brief      Get lceIdx that the PCE is mapped to
 *
 * @return     FALSE if there is no mapping
 *             TRUE and the lceIdx otherwise
 */
BOOL ceGetPceMap_GA100
(
    LwU32   pceIdx,
    LwU32  *pLceIdx
)
{
    LwU32 pce2LceConfigSize = pCe[indexGpu].ceGetPceToLceConfigSize();
    
    if (pceIdx < pce2LceConfigSize)
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
 * @details
 */
BOOL ceIsLceGrce_GA100(LwU32 lceIdx)
{
    if (lceIdx < LW_CE_GRCE_CONFIG__SIZE_1)
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

/**
 * @brief      Print all pce lce mapping info, and GRCE info
 */
void cePrintPceLceMap_GA100(void)
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
// ceDumpPriv_GV100 - Dumps CE priv reg space
//-----------------------------------------------------
LwU32 ceDumpPriv_GA100(LwU32 indexGpu, LwU32 indexCe)
{
    LwU32 u;
    LwU32 pcemap;
    char m_tag[256];

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u CE%d priv registers -- \n", indexGpu, indexCe);
    dprintf("lw:\n");

    // Global Registers
    sprintf(m_tag, "LW_CE_ENGCAP");
    cePrintPriv(40, m_tag, LW_CE_ENGCAP);
    sprintf(m_tag, "LW_CE_GRCE_CONFIG(0)");
    cePrintPriv(40, m_tag, LW_CE_GRCE_CONFIG(0));
    sprintf(m_tag, "LW_CE_GRCE_CONFIG(1)");
    cePrintPriv(40, m_tag, LW_CE_GRCE_CONFIG(1));
    dprintf("lw:\n");

    // INTR/BIND statuses, INTR EN, LAUNCHERR, THROTTLE, ENGCTL
    sprintf(m_tag, "LW_CE_LCE_INTR_EN(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_INTR_EN(indexCe));
    sprintf(m_tag, "LW_CE_LCE_INTR_STATUS(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_INTR_STATUS(indexCe));
    sprintf(m_tag, "LW_CE_LCE_BIND_STATUS(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_BIND_STATUS(indexCe));
    sprintf(m_tag, "LW_CE_LCE_LAUNCHERR(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_LAUNCHERR(indexCe));
    sprintf(m_tag, "LW_CE_LCE_THROTTLE(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_THROTTLE(indexCe));
    sprintf(m_tag, "LW_CE_LCE_ENGCTL(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_ENGCTL(indexCe));
    sprintf(m_tag, "LW_CE_LCE_OPT(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_OPT(indexCe));
    dprintf("lw:\n");

    // Get the mask of all enabled PCEs
    pcemap = GPU_REG_RD_DRF(_CE, _PCE_MAP, _VALUE);

    // Loop through all PCEs and read registers if the PCE is assigned to CE
    FOR_EACH_INDEX_IN_MASK(32, u, pcemap)
    {
        if(pCe[indexGpu].ceIsPceAssignedToLce(indexGpu, u, indexCe))
        {
            sprintf(m_tag, "LW_CE_PCE_CMD0(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_CMD0(u));
            sprintf(m_tag, "LW_CE_PCE_LINES_TO_COPY(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_LINES_TO_COPY(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM0(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM0(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM0(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM0(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM1(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM1(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM1(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM1(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM2(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM2(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM2(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM2(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM3(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM3(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM3(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM3(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM4(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM4(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM4(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM4(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PARAM5(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PARAM5(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PARAM5(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PARAM5(u));
            sprintf(m_tag, "LW_CE_PCE_SWIZZLE_CONSTANT0(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SWIZZLE_CONSTANT0(u));
            sprintf(m_tag, "LW_CE_PCE_SWIZZLE_CONSTANT1(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SWIZZLE_CONSTANT1(u));
            sprintf(m_tag, "LW_CE_PCE_CMD1(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_CMD1(u));
            sprintf(m_tag, "LW_CE_PCE_SRC_PHYS_MODE(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_SRC_PHYS_MODE(u));
            sprintf(m_tag, "LW_CE_PCE_DST_PHYS_MODE(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_DST_PHYS_MODE(u));
            sprintf(m_tag, "LW_CE_PCE_PIPESTATUS(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PIPESTATUS(u));
            sprintf(m_tag, "LW_CE_PCE_GLOBAL_COUNTER_LOWER(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_GLOBAL_COUNTER_LOWER(u));
            sprintf(m_tag, "LW_CE_PCE_GLOBAL_COUNTER_UPPER(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_GLOBAL_COUNTER_UPPER(u));
            sprintf(m_tag, "LW_CE_PCE_PAGEOUT_START_PALOWER(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PAGEOUT_START_PALOWER(u));
            sprintf(m_tag, "LW_CE_PCE_PAGEOUT_START_PAUPPER(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PAGEOUT_START_PAUPPER(u));
            sprintf(m_tag, "LW_CE_PCE_PMM(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PMM(u));
            dprintf("lw:\n");
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LW_OK;
}

//-----------------------------------------------------
// ceGetGrce_GP100: Return GRCE index
//-----------------------------------------------------
LwU32 ceGetGrce_GA100(LwU32 indexGpu)
{
    // No GRCOPY exists. Return max CE count from ptop
    return GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_CES, _VALUE);
}

//-----------------------------------------------------
// ceIsPresent_GA100
//-----------------------------------------------------
BOOL ceIsPresent_GA100(LwU32 indexCe)
{
    LwU32 i;

    //
    // CE fuses will be deprecated for Ampere+. Hence, look
    // into device info to determine if CE is present in ptop
    //
    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        for (i = 0; i < deviceInfo.enginesCount; ++i)
        {
            if ( (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENUM]
                     == LW_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE) &&
                 (deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INST_ID]
                     == indexCe) )
            {
                return TRUE;
            }
        }
    }

    return FALSE;
}

/*!
 *  Get the LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 *
 *  @return LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 */
LwU32 ceGetPceToLceConfigSize_GA100(void)
{
    return LW_CE_PCE2LCE_CONFIG__SIZE_1;
}

/*!
 *  Get the LW_CE_LCE_STATUS__SIZE_1 value
 *
 *  @return LW_CE_LCE_STATUS__SIZE_1 value
 */
LwU32 ceGetCeLceStatusSize_GA100(void)
{
    return LW_CE_LCE_STATUS__SIZE_1;
}