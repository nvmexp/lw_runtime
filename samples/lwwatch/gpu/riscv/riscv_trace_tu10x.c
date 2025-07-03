/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_riscv_pri.h"
#include "riscv_prv.h"

#include "g_riscv_private.h"

LW_STATUS riscvTraceEnable_TU10X(TRACE_MODE mode)
{
    bar0Write(LW_PRISCV_RISCV_TRACECTL,
              DRF_NUM(_PRISCV_RISCV, _TRACECTL, _MODE, mode) |
              DRF_DEF(_PRISCV_RISCV, _TRACECTL, _UMODE_ENABLE, _TRUE) |
              DRF_DEF(_PRISCV_RISCV, _TRACECTL, _MMODE_ENABLE, _TRUE) |
              DRF_DEF(_PRISCV_RISCV, _TRACECTL, _INTR_ENABLE, _FALSE) |
              DRF_DEF(_PRISCV_RISCV, _TRACECTL, _HIGH_THSHD, _INIT));
    return LW_OK;
}

LW_STATUS riscvTraceDisable_TU10X(void)
{
    bar0Write(LW_PRISCV_RISCV_TRACECTL, 0);
    return LW_OK;
}

LW_STATUS riscvTraceFlush_TU10X(void)
{
    LwU32 ctl = bar0Read(LW_PRISCV_RISCV_TRACECTL);

    // reset trace buffer
    bar0Write(LW_PRISCV_RISCV_TRACE_RDIDX, 0);
    bar0Write(LW_PRISCV_RISCV_TRACE_WTIDX, 0);

    // Clear full and empty bits
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _FULL, 0, ctl);
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _EMPTY, 0, ctl);
    bar0Write(LW_PRISCV_RISCV_TRACECTL, ctl);

    return LW_OK;
}

LW_STATUS riscvTraceDump_TU10X(void)
{
    LwU32 ctl, ridx, widx, count, bufferSize;
    LwBool full;

    ctl = bar0Read(LW_PRISCV_RISCV_TRACECTL);

    full = FLD_TEST_DRF_NUM(_PRISCV_RISCV,_TRACECTL,_FULL, 1, ctl);

    if (full)
        dprintf("Trace buffer full. Entries may have been lost.\n");

    // Reset and disable buffer, we don't need it during dump
    bar0Write(LW_PRISCV_RISCV_TRACECTL, 0);

    widx = DRF_VAL(_PRISCV_RISCV, _TRACE_WTIDX, _WTIDX, bar0Read(LW_PRISCV_RISCV_TRACE_WTIDX));

    ridx = bar0Read(LW_PRISCV_RISCV_TRACE_RDIDX);
    bufferSize = DRF_VAL(_PRISCV_RISCV, _TRACE_RDIDX, _MAXIDX, ridx);
    ridx = DRF_VAL(_PRISCV_RISCV, _TRACE_RDIDX, _RDIDX, ridx);

    count = widx > ridx ? widx - ridx : bufferSize + widx - ridx;

    //
    // Trace buffer is full when write idx == read idx and full is set,
    // otherwise it is empty.
    //
    if (widx == ridx && !full)
        count = 0;

    if (count)
    {
        LwU32 entry;
        dprintf("Tracebuffer has %d entries. Starting with latest.\n", count);
        ridx = widx;
        for (entry = 0; entry < count; ++entry)
        {
            LwU64 pc;

            ridx = ridx > 0 ? ridx - 1 : bufferSize - 1;
            bar0Write(LW_PRISCV_RISCV_TRACE_RDIDX, DRF_NUM(_PRISCV_RISCV, _TRACE_RDIDX, _RDIDX, ridx));
            pc = (((LwU64)bar0Read(LW_PRISCV_RISCV_TRACEPC_HI)) << 32) | bar0Read(LW_PRISCV_RISCV_TRACEPC_LO);
            dprintf("TRACE[%d] = "LwU64_FMT"\n", entry, pc);
        }
    } else
    {
        dprintf("Trace buffer is empty.\n");
    }

    // reset trace buffer
    bar0Write(LW_PRISCV_RISCV_TRACE_RDIDX, 0);
    bar0Write(LW_PRISCV_RISCV_TRACE_WTIDX, 0);

    // Clear full and empty bits
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _FULL, 0, ctl);
    ctl = FLD_SET_DRF_NUM(_PRISCV_RISCV, _TRACECTL, _EMPTY, 0, ctl);
    bar0Write(LW_PRISCV_RISCV_TRACECTL, ctl);

    return LW_OK;
}
