/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "gsp.h"
#include "lwsym.h"

#include "g_gsp_private.h"

#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_master.h"


// TODO: replace with proper define once it is known
#define GSP_MUTEX_EMEM 0

const char* gspGetEngineName_TU10X(void);
static const char* gspGetSymFilePath(void);

static FLCN_ENGINE_IFACES flcnEngineIfaces_gsp =
{
    gspGetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    gspGetFalconBase_STUB,              // flcnEngGetFalconBase
    gspGetEngineName_TU10X,             // flcnEngGetEngineName
    gspUcodeName_STUB,                  // flcnEngUcodeName
    gspGetSymFilePath,            // flcnEngGetSymFilePath
    gspQueueGetNum_STUB,                // flcnEngQueueGetNum
    gspQueueRead_STUB,                  // flcnEngQueueRead
    gspGetDmemAccessPort,               // flcnEngGetDmemAccessPort
    gspIsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    gspEmemGetSize_STUB,                // flcnEngEmemGetSize
    gspEmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    gspEmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    gspEmemRead_STUB,                   // flcnEngEmemRead
    gspEmemWrite_STUB,                  // flcnEngEmemWrite
};

const char* gspGetEngineName_TU10X(void)
{
    return "GSP";
}

static const char* gspGetSymFilePath(void)
{
    return DIR_SLASH "gsp" DIR_SLASH "bin";
}

LW_STATUS gspFillSymPath_TU10X(OBJFLCN *gspFlcn)
{
    sprintf(gspFlcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "gsp/");
    gspFlcn[indexGpu].bSympathSet = TRUE;
    return LW_OK;
}

LwU32 gspQueueGetNum_TU10X()
{
    return LW_PGSP_QUEUE_HEAD__SIZE_1 + LW_PGSP_MSGQ_HEAD__SIZE_1;
}

/*!
 *  Read the contents of a specific queue into queue. Message queue Id comes
 *  sequentially after the command queues. pQueue->id will be filled out
 *  automatically as well.
 *
 *  @param queueId Id of queue to get data for. If invalid, then this function
 *                 will return FALSE.
 *  @param pQueue  Pointer to queue structure to fill up.
 *
 *  @return FALSE if queueId is invalid or queue is NULL; TRUE on success.
 */
LwBool
gspQueueRead_TU10X
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pGsp[indexGpu].gspGetFalconEngineIFace();
    const FLCN_CORE_IFACES   *pFCIF =
        pGsp[indexGpu].gspGetFalconCoreIFace();
    LwU32 engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32 numQueues;
    LwU32 numCmdQs = LW_PGSP_QUEUE_HEAD__SIZE_1;
    LwU32 sizeInWords;
    LwU32 ememPort;

    numQueues = pGsp[indexGpu].gspQueueGetNum();
    if (queueId >= numQueues || !pQueue)
    {
        return LW_FALSE;
    }

    //
    // The "message" queues comes right after the command queues,
    // so we use a special case to get the information
    //
    if (queueId < LW_PGSP_QUEUE_HEAD__SIZE_1)
    {
        pQueue->head = GPU_REG_RD32(LW_PGSP_QUEUE_HEAD(queueId));
        pQueue->tail = GPU_REG_RD32(LW_PGSP_QUEUE_TAIL(queueId));
    }
    else
    {
        pQueue->head = GPU_REG_RD32(LW_PGSP_MSGQ_HEAD(queueId-numCmdQs));
        pQueue->tail = GPU_REG_RD32(LW_PGSP_MSGQ_TAIL(queueId-numCmdQs));
    }

    //
    // At the momement, we assume that tail <= head. This is not the case since
    // the queue wraps-around. Unfortunatly, we have no way of knowing the size
    // or offsets of the queues, and thus renders the parsing slightly
    // impossible. Lwrrently do not support.
    //
    if (pQueue->head < pQueue->tail)
    {
        dprintf("lw: Queue 0x%x is lwrrently in a wrap-around state.\n",
                queueId);
        dprintf("lw:     tail=0x%04x, head=0x%04x\n", pQueue->tail, pQueue->head);
        dprintf("lw:     It is lwrrently not possible to parse this queue.\n");
        return LW_FALSE;
    }

    sizeInWords    = (pQueue->head - pQueue->tail) / sizeof(LwU32);
    pQueue->length = sizeInWords;
    pQueue->id     = queueId;

    //
    // If the queue happens to be larger than normally allowed, print out an
    // error message and return an error.
    //
    if (sizeInWords >= LW_FLCN_MAX_QUEUE_SIZE)
    {
        dprintf("lw: %s: GSP queue 0x%x is larger than configured to read:\n",
                __FUNCTION__, queueId);
        dprintf("lw:     Queue Size: 0x%x     Supported Size: 0x%x\n",
                (LwU32)(sizeInWords * sizeof(LwU32)), (LwU32)(LW_FLCN_MAX_QUEUE_SIZE * sizeof(LwU32)));
        dprintf("lw:     Make LW_FLCN_MAX_QUEUE_SIZE larger and re-compile LW_WATCH\n");
        return LW_FALSE;
    }

    ememPort = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;

    // Simply read the queue into the buffer if it is initialized
    if (pQueue->tail > pFEIF->flcnEngEmemGetOffsetInDmemVaSpace())
    {
        pFEIF->flcnEngEmemRead(pQueue->tail, sizeInWords, ememPort,
                               pQueue->data);
    }
    return LW_TRUE;
}

LwU32 gspEmemGetNumPorts_TU10X(void)
{
    return LW_PGSP_EMEMD__SIZE_1;
}

const FLCN_ENGINE_IFACES *gspGetFalconEngineIFace_TU10X(void)
{
    const FLCN_CORE_IFACES   *pFCIF = pGsp[indexGpu].gspGetFalconCoreIFace();
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_gsp;

        pFEIF->flcnEngGetCoreIFace               = pGsp[indexGpu].gspGetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase              = pGsp[indexGpu].gspGetFalconBase;
        pFEIF->flcnEngQueueGetNum                = pGsp[indexGpu].gspQueueGetNum;
        pFEIF->flcnEngQueueRead                  = pGsp[indexGpu].gspQueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible      = pGsp[indexGpu].gspIsDmemRangeAccessible;
        pFEIF->flcnEngEmemGetSize                = pGsp[indexGpu].gspEmemGetSize;
        pFEIF->flcnEngEmemGetOffsetInDmemVaSpace = pGsp[indexGpu].gspEmemGetOffsetInDmemVaSpace;
        pFEIF->flcnEngEmemGetNumPorts            = pGsp[indexGpu].gspEmemGetNumPorts;
        pFEIF->flcnEngEmemRead                   = pGsp[indexGpu].gspEmemRead;
        pFEIF->flcnEngEmemWrite                  = pGsp[indexGpu].gspEmemWrite;
    }
    return pFEIF;
}
