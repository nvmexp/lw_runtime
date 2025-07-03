/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "fifo.h"
#include "mmu.h"

#include "pascal/gp102/dev_ram.h"
#include "pascal/gp102/dev_fifo.h"
#include "g_fifo_private.h"

LwU32 fifoGetNumEng_GP102(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

void fifoPrintRunlistEntry_GP102(LwU32 unused, LwU32 *entry)
{
    if (!entry)
    {
        dprintf("Invalid entry");
        return;
    }

    fifoPrintRunlistEntry_GK104(unused, entry);

    if(DRF_VAL(_RAMRL, _ENTRY, _TYPE, entry[0]) == LW_RAMRL_ENTRY_TYPE_CHID)
    {
        LwU32 runqueue = DRF_VAL(_RAMRL, _ENTRY, _RUNQUEUE_SELECTOR, entry[0]);
        dprintf("       RunQ=0x%-3x", runqueue);
    }
}
