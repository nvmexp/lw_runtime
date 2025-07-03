/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <utils/lwassert.h>

#include "riscv_prv.h"

#include "hal.h"
#include "g_riscv_hal.h"

#include "riscv_config.h"

LwU64 triggerAddrs[TRIGGERS_MAX] = {0, };
TRIGGER_EVENT triggerEvents[TRIGGERS_MAX] = {TRIGGER_UNUSED, };

LW_STATUS riscvTriggerClearAll(void)
{
    int i;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Target must be halted.\n");
        return LW_ERR_ILWALID_STATE;
    }

    for (i=0; i<TRIGGERS_MAX; ++i)
    {
        if (triggerEvents[i] != TRIGGER_UNUSED)
        {
            triggerAddrs[i] = 0;
            triggerEvents[i] = TRIGGER_UNUSED;
            CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu]._triggerWrite(i, 0, 0));
        }
    }
    return LW_OK;
}
