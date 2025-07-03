
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2020 by LWPU Corporation.  All rights reserved.  All
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
#include "vmem.h"
#include "maxwell/gm200/hwproject.h"
#include "maxwell/gm200/dev_fifo.h"
#include "maxwell/gm200/dev_ram.h"
#include "maxwell/gm200/dev_pbdma.h"

/*!
 * @return The number of engines provided by the chip.
 */
LwU32 fifoGetNumEng_GM200(void)
{
    return LW_PFIFO_ENGINE_STATUS__SIZE_1;
}

/*!
 * @return The number of PBDMAs provided by the chip.
 */
LwU32 fifoGetNumPbdma_GM200(void)
{
    return LW_HOST_NUM_PBDMA;
}

/*!
 * @return The maximum number of channels provided by the chip
 */
LwU32 fifoGetNumChannels_GM200(LwU32 runlistId)
{
    // Unused pre-Ampere
    (void) runlistId;

    return LW_PCCSR_CHANNEL__SIZE_1;
}

/*!
 * Dumps the SCG info
 */
void fifoDumpSubctxInfo_GM200(ChannelId *pChannelId)
{
    LwU32    buf, scgType;
    LwU64    instMemAddr;
    readFn_t readFn = NULL;

    // get instance memory address for this channel
    instMemAddr = pVmem[indexGpu].vmemGetInstanceMemoryAddrForChId(pChannelId, &readFn, NULL, NULL);
    if (readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return;
    }

    readFn(instMemAddr + SF_OFFSET(LW_RAMFC_SET_CHANNEL_INFO), &buf, 4);
    scgType = DRF_VAL(_PPBDMA, _SET_CHANNEL_INFO, _SCG_TYPE, buf);
    dprintf(" + SCG TYPE:                           %d\n", scgType);
}
