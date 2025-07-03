/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file    flcnrtosdebug_lwswitch.c
 * @brief   Provides support for capturing RTOS's state in case of Falcon
 *          related failures.
 */

/* ------------------------- Includes --------------------------------------- */

#include "common_lwswitch.h"

#include "flcn/flcn_lwswitch.h"
#include "flcn/flcnable_lwswitch.h"
#include "rmflcncmdif_lwswitch.h"

#include "flcn/flcnrtosdebug_lwswitch.h"

/*!
 * Dump the complete stack by iterating from tail to head pointer
 *
 *  @param[in]  device        lwswitch_device pointer
 *  @param[in]  pFlcn         FLCN pointer
 *  @param[in]  queueLogId    Logical ID of the queue
 *  @param[in]  pFlcnCmd      Pointer to the command buffer to read
 *
 */
LW_STATUS
flcnRtosDumpCmdQueue_lwswitch
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            queueLogId,
    RM_FLCN_CMD     *pFlcnCmd
)
{
    FLCNQUEUE  *pQueue;
    LwU32      head;
    LwU32      tail;
    LwU32      tailcache;

    LW_STATUS  status             = LW_OK;
    PFALCON_QUEUE_INFO pQueueInfo = pFlcn->pQueueInfo;

    pQueue = &pQueueInfo->pQueues[queueLogId];
    (void)pQueue->tailGet(device, pFlcn, pQueue, &tail);
    (void)pQueue->headGet(device, pFlcn, pQueue, &head);

    // caching the current tail pointer
    (void)pQueue->tailGet(device, pFlcn, pQueue, &tailcache);

    if (head == tail)
    {
        return status;
    }

    while (tail != head)
    {
        status = flcnQueueReadData(device,pFlcn,
                                    queueLogId,
                                    pFlcnCmd, LW_FALSE);
        LWSWITCH_PRINT(device, ERROR, "%s:" \
                    "Cmd_Dump UnitId %d size %d sq %d ctl %d cmd %d\n",
                    __FUNCTION__,
                    pFlcnCmd->cmdGen.hdr.unitId,
                    pFlcnCmd->cmdGen.hdr.size,
                    pFlcnCmd->cmdGen.hdr.seqNumId,
                    pFlcnCmd->cmdGen.hdr.ctrlFlags,
                    pFlcnCmd->cmdGen.cmd);

        (void)pQueue->tailGet(device, pFlcn, pQueue, &tail);
    }

    // restoring the cached tail pointer
    (void)pQueue->tailSet(device, pFlcn, pQueue, tailcache);

    return status;
}

/*!
 * @brief Populates falcon DMEM pointer in its internal debug info structure
 *
 * @param[in]  device               GPU object pointer
 * @param[in]  pFlcn                FLCN pointer
 * @param[in]  debugInfoDmemOffset  DMEM offset of the falcon debug info
 */
static void
_flcnDbgInfoDmemOffsetSet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU16            debugInfoDmemOffset
)
{
    pFlcn->debug.debugInfoDmemOffset = debugInfoDmemOffset;
}

void
flcnRtosSetupHal
(
    FLCN   *pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    pHal->dbgInfoDmemOffsetSet  = _flcnDbgInfoDmemOffsetSet_IMPL;
}

