/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// adimitrov@lwpu.com - July 20 2007
// vmemgm107.c - page table routines for Fermi
//
//*****************************************************

//
// Includes
//

#include "fb.h"
#include "vmem.h"

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForChId_GM107
//
// Returns the instance memory base address for a
// channel ID. It works for DMA mode.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForChId_GM107(ChannelId *pChannelId, readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LW_STATUS   status = LW_OK;
    LwU64       instMemAddr = 0;
    LwU64       instMemTarget = 0;
    LwU64       targetMemType;
    ChannelInst channelInst;

    pFifo[indexGpu].fifoGetChannelInstForChid(pChannelId, &channelInst);
    instMemAddr = channelInst.instPtr;
    instMemTarget = channelInst.target;

    status = pVmem[indexGpu].vmemGetMemTypeFromTarget(instMemTarget, &targetMemType);

    if ( status != LW_OK )
    {
        dprintf("lwwatch: %s: Failed to determine memory type from instance memory target\n", __FUNCTION__);
    }

    if (pMemType)
        *pMemType = (MEM_TYPE)targetMemType;

    switch (targetMemType)
    {
    case FRAMEBUFFER:
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        break;

    case SYSTEM_PHYS:
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        break;
    }

    return instMemAddr;
}
