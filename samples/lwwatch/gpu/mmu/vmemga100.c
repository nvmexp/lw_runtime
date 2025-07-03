/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "fifo.h"
#include "ampere/ga100/dev_ram.h"

#include "g_vmem_private.h"

LW_STATUS
vmemInitLayout_GA100(VMemSpace *pVMemSpace)
{
    LW_STATUS   retval;

    retval = vmemInitLayout_GP100(pVMemSpace);

    // Only difference is that GA100 supports 512MB pages
    pVMemSpace->layout.fmtLevels[2].bPageTable = LW_TRUE;

    return retval;
}

//-----------------------------------------------------
// Colwerts an instance memory target to a MEM_TYPE.
// @param[in] instMemTarget
// @param[out] pMemType
//-----------------------------------------------------
LW_STATUS
vmemGetMemTypeFromTarget_GA100(LwU64 instMemTarget, LwU64* pMemType)
{
    LW_STATUS status = LW_OK;
    MEM_TYPE memType = SYSTEM_PHYS;

    switch (instMemTarget)
    {
        case LW_RAMRL_ENTRY_CHAN_INST_TARGET_VID_MEM:
            memType = FRAMEBUFFER;
            break;

        case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_COHERENT:
        case LW_RAMRL_ENTRY_CHAN_INST_TARGET_SYS_MEM_NONCOHERENT:
            memType = SYSTEM_PHYS;
            break;

        default:
            status = LW_ERR_GENERIC;
    }

    if (status == LW_OK && pMemType)
    {
        *pMemType = (LwU64)memType;
    }

    return status;
}
