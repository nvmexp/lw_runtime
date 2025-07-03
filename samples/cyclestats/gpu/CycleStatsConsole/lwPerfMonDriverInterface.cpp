 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <windows.h>

#include <lw32.h>
#include <lwos.h>
#include <lwcm.h>
#include <lwrmapi.h>

#include "lwtypes.h"
#include "Lwcm.h"

#include "CycleStatsConsole.h"
#include "../../../../drivers/common/cyclestats/gpu/lwPerfMonHW.h"

union UPointer
{
    void           *v;
    volatile LwU32 *u32;
    volatile LwU64 *u64;
};

bool PMDriverInterface::identifyChip(void *context, LwU32 *pArch, LwU32 *pImpl)
{
    // context is the chip id :)
    unsigned int chipId = (unsigned int)(size_t)context;
    LwU32 arch = chipId & 0xfffffff0;
    LwU32 impl = chipId & 0x0000000f;
    if (*pArch)
    {
        *pArch = arch;
    }
    if (*pImpl)
    {
        *pImpl = impl;
    }
    return true;
}

void PMDriverInterface::getPerfmonBufferInfo(void *context, LWStreamId streamId, void **ppFIFO, LwU32 *pPMTriggerCount, LwU32 *pTimerBase, LwU32 *pPerfmonIdStart, LwU32 *pPerfmonIdCount, LwU32 *pUsePTIMER)
{
    // not supported (no snapshot buffer support implemented)
    *ppFIFO = NULL;
    *pPMTriggerCount = 0;
    *pTimerBase = 0;
    *pPerfmonIdStart = 0;
    *pPerfmonIdCount = 0;
    *pUsePTIMER = false;
}

void PMDriverInterface::push1U(void *context, LWStreamId streamId, LwU32 hwMethod, LwU32 hwData, LwU32 gpuMask)
{
    // not supported (no pushbuffer implemented)
    assert(0);
}

void PMDriverInterface::pushBAR0Write32(void *context, LWStreamId streamIdMask, LwU32 offsetBAR0, LwU16 firstBit, LwU16 lastBit, LwU32 data)
{
    if (g_accessBAR0ViaRM)
    {
        assert(0);
    }
    else
    {
        UPointer p = {(char*)g_bar0 + offsetBAR0};
        LwU32 mask = (LwU32) (((1ULL<<(lastBit+1))-1) & ~((1ULL<<firstBit)-1));
        LwU32 v = 0;

        // avoid useless reads (after all they are really slow)
        if ((LwU32)mask != ~0)
        {
            v = (LwU32) (*p.u32 & ~mask);
        }
        v |= ((data << firstBit) & mask);
        *p.u32 = (unsigned int)v;
    }
}

void PMDriverInterface::pushBAR0Write64(void *context, LWStreamId streamIdMask, LwU32 offsetBAR0, LwU16 firstBit, LwU16 lastBit, LwU64 data)
{
    if (g_accessBAR0ViaRM)
    {
        assert(0);
    }
    else
    {
        UPointer p = {(char*)g_bar0 + offsetBAR0};
        LwU64 mask = ((1ULL<<(lastBit+1))-1) & ~((1ULL<<firstBit)-1);
        LwU64 v = 0;
        
        // avoid useless reads (after all they are really slow)
        if (mask != ~0ULL)
        {
            v = (*p.u64 & ~mask);
        }
        v |= ((data << firstBit) & mask);
        *p.u64 = v;
    }
}

void PMDriverInterface::pushBAR0Read32(void *context, LWStreamId streamId, LwU32 *pDone, LwU32 *pData, LwU32 offsetBAR0, LwU16 firstBit, LwU16 lastBit)
{
    if (g_accessBAR0ViaRM)
    {
        assert(0);
    }
    else
    {
        UPointer p = {(char*)g_bar0 + offsetBAR0};
        LwU32 mask = (LwU32) (((1ULL<<(lastBit+1))-1) & ~((1ULL<<firstBit)-1));
        *pData = (*p.u32 & mask) >> firstBit;
        if (pDone)
        {
            *pDone = true;
        }
    }
}

void PMDriverInterface::pushBAR0Read64(void *context, LWStreamId streamId, LwU32 *pDone, LwU64 *pData, LwU32 offsetBAR0, LwU16 firstBit, LwU16 lastBit)
{
    if (g_accessBAR0ViaRM)
    {
        assert(0);
    }
    else
    {
        UPointer p = {(char*)g_bar0 + offsetBAR0};
        LwU64 mask = ((1ULL<<(lastBit+1))-1) & ~((1ULL<<firstBit)-1);
        *pData = (*p.u64 & mask) >> firstBit;
        if (pDone)
        {
            *pDone = true;
        }
    }
}

void PMDriverInterface::getQueryBufferInfo(void *context, void **ppCpuBaseAddr, LwOffset *pGpuBaseOffset, LwU32 *pBufferSize, bool *pHasQueryGrayCodeBug)
{
    // not supported - use a fake/dummy allocation to silence assertions
    static char unusedScratchSpace[16*1024];
    if (ppCpuBaseAddr)
    {
        *ppCpuBaseAddr = unusedScratchSpace;
    }
    if (pGpuBaseOffset)
    {
        *pGpuBaseOffset = 0;
    }
    if (pBufferSize)
    {
        // dummy allocation
        *pBufferSize = sizeof(unusedScratchSpace);
    }
    if (pHasQueryGrayCodeBug)
    {
        *pHasQueryGrayCodeBug = 0;
    }
}

void PMDriverInterface::pushQuery(void *context, LWStreamId streamId, LwOffset hwQueryOffset, LwU32 hwType, LwU32 gpuMask)
{
    // not supported (no pushbuffer implemented)
    assert(0);
}

void PMDriverInterface::finish(void *context, LWStreamId streamId)
{
    // no ops are deferred
}
