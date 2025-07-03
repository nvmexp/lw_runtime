/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "pascal/gp102/dev_top.h"
#include "pascal/gp102/dev_ce.h"

#include "ce.h"
#include "hal.h"
#include "g_ce_hal.h"

//-----------------------------------------------------
// ceDumpPriv_GP102 - Dumps CE priv reg space
//-----------------------------------------------------
LwU32 ceDumpPriv_GP102( LwU32 indexGpu, LwU32 indexCe )
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
