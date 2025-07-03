/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "fbflcn.h"

#include "g_fbflcn_private.h"

#include "volta/gv100/dev_fbfalcon_pri.h"
#include "volta/gv100/dev_master.h"


static FLCN_ENGINE_IFACES flcnEngineIfaces_fbflcn =
{
    fbflcnGetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    fbflcnGetFalconBase_STUB,              // flcnEngGetFalconBase
    fbflcnGetEngineName,                   // flcnEngGetEngineName
    fbflcnUcodeName_STUB,                  // flcnEngUcodeName
    fbflcnGetSymFilePath,                  // flcnEngGetSymFilePath
    fbflcnQueueGetNum_STUB,                // flcnEngQueueGetNum
    fbflcnQueueRead_STUB,                  // flcnEngQueueRead
    fbflcnGetDmemAccessPort,               // flcnEngGetDmemAccessPort
    fbflcnIsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    fbflcnEmemGetSize_STUB,                // flcnEngEmemGetSize
    fbflcnEmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    fbflcnEmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    fbflcnEmemRead_STUB,                   // flcnEngEmemRead
    fbflcnEmemWrite_STUB,                  // flcnEngEmemWrite
};

LwBool fbflcnIsSupported_GV100(void)
{
    return TRUE;
}

LwU32 fbflcnQueueGetNum_GV100()
{
    return LW_PFBFALCON_CMD_QUEUE_HEAD__SIZE_1 + LW_PFBFALCON_MSGQ_HEAD__SIZE_1;
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
fbflcnQueueRead_GV100
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    const FLCN_ENGINE_IFACES *pFEIF =
        pFbflcn[indexGpu].fbflcnGetFalconEngineIFace();
    const FLCN_CORE_IFACES   *pFCIF =
        pFbflcn[indexGpu].fbflcnGetFalconCoreIFace();
    LwU32 engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32 numQueues;
    LwU32 numCmdQs = LW_PFBFALCON_CMD_QUEUE_HEAD__SIZE_1;
    LwU32 sizeInWords;
    LwU32 ememPort;

    numQueues = pFbflcn[indexGpu].fbflcnQueueGetNum();
    if (queueId >= numQueues || (pQueue == NULL))
    {
        return LW_FALSE;
    }

    //
    // The "message" queues comes right after the command queues,
    // so we use a special case to get the information
    //
    if (queueId < LW_PFBFALCON_CMD_QUEUE_HEAD__SIZE_1)
    {
        pQueue->head = GPU_REG_RD32(LW_PFBFALCON_CMD_QUEUE_HEAD(queueId));
        pQueue->tail = GPU_REG_RD32(LW_PFBFALCON_CMD_QUEUE_TAIL(queueId));
    }
    else
    {
        pQueue->head = GPU_REG_RD32(LW_PFBFALCON_MSGQ_HEAD(queueId-numCmdQs));
        pQueue->tail = GPU_REG_RD32(LW_PFBFALCON_MSGQ_TAIL(queueId-numCmdQs));
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
        dprintf("lw: %s: FBFalcon queue 0x%x is larger than configured to read:\n",
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

const FLCN_ENGINE_IFACES *fbflcnGetFalconEngineIFace_GV100(void)
{
    const FLCN_CORE_IFACES   *pFCIF = pFbflcn[indexGpu].fbflcnGetFalconCoreIFace();
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_fbflcn;

        pFEIF->flcnEngGetCoreIFace               = pFbflcn[indexGpu].fbflcnGetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase              = pFbflcn[indexGpu].fbflcnGetFalconBase;
        pFEIF->flcnEngQueueGetNum                = pFbflcn[indexGpu].fbflcnQueueGetNum;
        pFEIF->flcnEngQueueRead                  = pFbflcn[indexGpu].fbflcnQueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible      = pFbflcn[indexGpu].fbflcnIsDmemRangeAccessible;
    }
    return pFEIF;
}

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
fbflcnGetFalconCoreIFace_GV100()
{
    return &flcnCoreIfaces_v06_00;
}

/*!
 * @return Get the number of DMEM carveouts
 */
LwU32 fbflcnGetDmemNumPrivRanges_GV100(void)
{
    // No register to read this, so hardcode until HW adds support.
    return 2; // RANGE0/1
}

/*!
 * @return Get the DMEM Priv Range0/1
 */
void
fbflcnGetDmemPrivRange_GV100
(
    LwU32  index,
    LwU32 *rangeStart,
    LwU32 *rangeEnd
)
{
    LwU32 reg;

    switch(index)
    {
        case 0:
        {
            reg         = GPU_REG_RD32(LW_PFBFALCON_FALCON_DMEM_PRIV_RANGE0);
            *rangeStart = DRF_VAL(_PFBFALCON_FALCON, _DMEM_PRIV_RANGE0, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PFBFALCON_FALCON, _DMEM_PRIV_RANGE0, _END_BLOCK, reg);
            break;
        }

        case 1:
        {
            reg         = GPU_REG_RD32(LW_PFBFALCON_FALCON_DMEM_PRIV_RANGE1);
            *rangeStart = DRF_VAL(_PFBFALCON_FALCON, _DMEM_PRIV_RANGE1, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PFBFALCON_FALCON, _DMEM_PRIV_RANGE1, _END_BLOCK, reg);
            break;
        }

        default:
        {
            *rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
            *rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
            dprintf("lw: Invalid index: %d\n", index);
            break;
        }
    }
}

/*!
 * @return LW_TRUE  DMEM range is accessible
 *         LW_FALSE DMEM range is inaccessible
 */
LwBool
fbflcnIsDmemRangeAccessible_GV100
(
    LwU32 blkLo,
    LwU32 blkHi
)
{
    LwU32  i, numPrivRanges;
    LwU32  rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwU32  rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwBool accessAllowed = LW_FALSE;

    numPrivRanges = pFbflcn[indexGpu].fbflcnGetDmemNumPrivRanges();

    for (i = 0; i < numPrivRanges; i++)
    {
        pFbflcn[indexGpu].fbflcnGetDmemPrivRange(i, &rangeStart, &rangeEnd);

        if (rangeStart >= rangeEnd)
        {
            // invalid range.
            continue;
        }

        if (blkLo >= rangeStart && blkHi <= rangeEnd)
        {
            // We're within range
            accessAllowed = LW_TRUE;
            break;
        }
    }

    if (!accessAllowed)
    {
        // Print out info message
        dprintf("lw: FBFalcon is in LS mode. Requested address range is not within "
                "ranges accessible by CPU.\n");
    }

    return accessAllowed;
}

/*!
 * @return The falcon base address of FBFalcon
 */
LwU32
fbflcnGetFalconBase_GV100()
{
    return DEVICE_BASE(LW_PFBFALCON);
}


