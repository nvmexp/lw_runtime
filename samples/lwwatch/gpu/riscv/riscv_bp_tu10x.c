/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <utils/lwassert.h>

#include "riscv_prv.h"
#include "turing/tu102/dev_gsp_riscv_csr_64.h"

#include "g_riscv_private.h"

LW_STATUS _triggerWrite_TU10X(LwU8 tselect, LwU64 tdata1, LwU64 tdata2)
{
    LwU64 v = ~tselect;

    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, tselect));

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TSELECT, &v));

    if (v != tselect)
    {
        dprintf("Failed to select trigger %d\n", tselect);
        return LW_ERR_OBJECT_NOT_FOUND;
    }

    //
    // When setting breakpoint, write address first,
    // When clearing breakpoint, disable breakpoint first.
    //
    if (tdata1 != 0)
        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA2, tdata2));
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA1, tdata1));
    if (tdata1 == 0)
        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA2, tdata2));

    return LW_OK;
}

LW_STATUS riscvTriggerSetAt_TU10X(LwU64 addr, TRIGGER_EVENT event)
{
    int i;

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Target must be halted.\n");
        return LW_ERR_ILWALID_STATE;
    }

    LW_ASSERT_OR_RETURN(event <= 0x7, LW_ERR_ILWALID_ARGUMENT);
    LW_ASSERT_OR_RETURN(event != TRIGGER_UNUSED, LW_ERR_ILWALID_ARGUMENT);

    for (i=0; i<TRIGGERS_MAX; ++i)
    {
        /*
         * Use trigger if:
         * - It is unused, or
         * - It is used for different type of events
         * - Do not permit to set the same trigger twice
         */

        if ((triggerAddrs[i] == addr) && ((triggerEvents[i] & event) != TRIGGER_UNUSED))
        {
            dprintf("Can't set the same trigger twice.\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }

        if ((triggerEvents[i] == TRIGGER_UNUSED) || (triggerAddrs[i] == addr))
        {
            triggerAddrs[i] = addr;
            triggerEvents[i] |= event;
            return _triggerWrite_TU10X(i,
                                       triggerEvents[i] |
                                       DRF_DEF(_RISCV, _CSR_TDATA1, _U, _ENABLE) |
                                       DRF_DEF(_RISCV, _CSR_TDATA1, _M, _ENABLE) |
                                       DRF_DEF(_RISCV, _CSR_TDATA1, _ACTION, _ICD),
                                       addr);
        }
    }

    return LW_ERR_INSUFFICIENT_RESOURCES;
}

LW_STATUS riscvTriggerClearAt_TU10X(LwU64 addr, TRIGGER_EVENT event)
{
    int i;

    for (i=0; i<TRIGGERS_MAX; ++i)
    {
        if (triggerAddrs[i] == addr)
        {
            if (triggerEvents[i] != TRIGGER_UNUSED)
            {
                triggerEvents[i] = triggerEvents[i] & ~event;
                if (triggerEvents[i] == TRIGGER_UNUSED)
                {
                    triggerAddrs[i] = 0;
                    CHECK_SUCCESS_OR_RETURN(_triggerWrite_TU10X(i, 0, 0));
                }
                else
                {
                    return _triggerWrite_TU10X(i,
                                               triggerEvents[i] |
                                               DRF_DEF(_RISCV, _CSR_TDATA1, _U, _ENABLE) |
                                               DRF_DEF(_RISCV, _CSR_TDATA1, _M, _ENABLE) |
                                               DRF_DEF(_RISCV, _CSR_TDATA1, _ACTION, _ICD),
                                               addr);
                }
            }
            return LW_OK; // Only one trigger can be set
        }
    }

    return LW_ERR_OBJECT_NOT_FOUND;
}
