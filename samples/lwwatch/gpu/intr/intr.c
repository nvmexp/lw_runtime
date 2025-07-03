/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "chip.h"
#include "hal.h"
#include "intr.h"
#include "lwwatch.h"
#include "os.h"
#include "print.h"
#include "vgpu.h"

#include "g_intr_private.h"

static LwBool bInterruptInit = LW_FALSE;

void intrPrintHelp(void)
{
    dprintf("Interrupt Routines:\n");
    dprintf(" -enum                             - Enumerates all the interrupts that report into the GPU interrupt tree.\n");
    dprintf(" -status <cpu/gsp> -gfid <gfid>    - Lists all pending interrupts with unit info and vector number.\n");
    dprintf("                                     This will also dump leaf and top register info for the gfid.\n");
    dprintf(" -set <vect> -gfid <gfid>          - Sets the specified interrupt for a gfid, if possible.\n");
    dprintf("                                     Setting an interrupt in the GSP tree is not supported.\n");
    dprintf(" -clear <vect> -gfid <gfid>        - Clears the specified interrupt for a gfid, if possible.\n");
    dprintf("                                     Clearing an interrupt in the GSP tree is not supported.\n");
    dprintf(" -enable <vect> -gfid <gfid>       - Enables the interrupt vector at the INTR_LEAF and INTR_TOP levels in the GPU interrupt tree.\n");
    dprintf(" -disable <vect> -gfid <gfid>      - Clears the enable for the interrupt vector at the INTR_LEAF level.\n");
    dprintf("\n\n");
    return;
}

/**
 * @brief   Enumerates all interrupts, including those not pending
 */
void intrEnum(void)
{
    if (!bInterruptInit)
    {
        intrEnumTable.bInterruptTableInit = LW_FALSE;
        bInterruptInit = LW_TRUE;
    }

    if (!intrEnumTable.bInterruptTableInit)
    {
        pIntr[indexGpu].intrInit();
        intrEnumTable.bInterruptTableInit = LW_TRUE;
    }

    pIntr[indexGpu].intrPrintList(LW_FALSE, INTR_DEFAULT_GFID, LW_FALSE);
    return;
}

/**
 * @brief   Enumerates all pending interrupts
 * @param   idx     if -1, then GSP tree, else a gfid index
 * @param   bGsp    whether or not target is gsp
 */
void intrEnumPending(LwU32 gfid, LwBool bGsp)
{
    if (!bInterruptInit)
    {
        intrEnumTable.bInterruptTableInit = LW_FALSE;
        bInterruptInit = LW_TRUE;
    }

    if (!intrEnumTable.bInterruptTableInit)
    {
        pIntr[indexGpu].intrInit();
        intrEnumTable.bInterruptTableInit = LW_TRUE;
    }

    if (gfid >= MAX_GFIDS && !bGsp)
    {
        dprintf("lw: Error: Invalid function ID.\n");
        return;
    }

    dprintf("lw: Pending interrupts:\n");
    pIntr[indexGpu].intrPrintList(LW_TRUE, gfid, bGsp);
    return;
}
