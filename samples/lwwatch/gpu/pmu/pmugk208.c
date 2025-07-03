/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "pmu.h"
#include "kepler/gk208/dev_pwr_pri.h"

/*!
 * Read LW_PPWR_PMU_MAILBOX(i)
 */
LwU32
pmuReadPmuMailbox_GK208
(
    LwU32 index
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_MAILBOX(index));
}

/*!
 * Writes to LW_PPWR_PMU_MAILBOX(i)
 */
void
pmuWritePmuMailbox_GK208
(
    LwU32 index,
    LwU32 value
)
{
    PMU_REG_WR32(LW_PPWR_PMU_MAILBOX(index), value);
    return;
}

/*!
 * Reads LW_PPWR_PMU_NEW_INSTBLK
 */
LwU32 pmuReadPmuNewInstblk_GK208()
{
    return PMU_REG_RD32(LW_PPWR_PMU_NEW_INSTBLK);
}

const char *
pmuUcodeName_GK208()
{
    return "g_c85b6_gk20x";
}
