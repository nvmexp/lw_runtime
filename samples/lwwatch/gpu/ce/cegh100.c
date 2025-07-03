/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "hopper/gh100/dev_top.h"
#include "hopper/gh100/dev_ce.h"

#include "hwref/lwutil.h"
#include "fifo.h"
#include "deviceinfo.h"
#include "g_ce_hal.h"
#include "ce.h"

/**
 * @brief      Get lceIdx that the PCE is mapped to
 *
 * @return     FALSE if there is no mapping
 *             TRUE and the lceIdx otherwise
 */
BOOL ceGetPceMap_GH100
(
    LwU32   pceIdx,
    LwU32  *pLceIdx
)
{
    if (pceIdx < LW_CE_PCE2LCE_CONFIG__SIZE_1)
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

//-----------------------------------------------------
// ceDumpPriv_GV100 - Dumps CE priv reg space
//-----------------------------------------------------
LwU32 ceDumpPriv_GH100(LwU32 indexGpu, LwU32 indexCe)
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
    sprintf(m_tag, "LW_CE_LCE_BIND_STATUS_UPPER(%d)", indexCe);
    cePrintPriv(40, m_tag, LW_CE_LCE_BIND_STATUS_UPPER(indexCe));
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
            sprintf(m_tag, "LW_CE_PCE_PIPESTATUS(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PIPESTATUS(u));
            sprintf(m_tag, "LW_CE_PCE_PMM(%d)", u);
            cePrintPriv(40, m_tag, LW_CE_PCE_PMM(u));
            dprintf("lw:\n");
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LW_OK;
}

/*!
 *  Get the LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 *
 *  @return LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 */
LwU32 ceGetPceToLceConfigSize_GH100(void)
{
    return LW_CE_PCE2LCE_CONFIG__SIZE_1;
}