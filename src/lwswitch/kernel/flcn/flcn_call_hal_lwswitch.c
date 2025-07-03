/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "flcn/haldefs_flcn_lwswitch.h"
#include "flcn/flcn_lwswitch.h"

#include "flcnifcmn.h"

typedef union  RM_FLCN_CMD RM_FLCN_CMD, *PRM_FLCN_CMD;
typedef union  RM_FLCN_MSG RM_FLCN_MSG, *PRM_FLCN_MSG;

// OBJECT Interfaces
LW_STATUS
flcnQueueReadData
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            queueId,
    void            *pData,
    LwBool           bMsg
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueReadData != (void *)0);
    return pFlcn->pHal->queueReadData(device, pFlcn, queueId, pData, bMsg);
}

LW_STATUS
flcnQueueCmdWrite
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            queueId,
    RM_FLCN_CMD     *pCmd,
    LWSWITCH_TIMEOUT *pTimeout
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueCmdWrite != (void *)0);
    return pFlcn->pHal->queueCmdWrite(device, pFlcn, queueId, pCmd, pTimeout);
}

LW_STATUS
flcnQueueCmdCancel
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            seqDesc
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueCmdCancel != (void *)0);
    return pFlcn->pHal->queueCmdCancel(device, pFlcn, seqDesc);
}

LW_STATUS
flcnQueueCmdPostNonBlocking
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PRM_FLCN_CMD     pCmd,
    PRM_FLCN_MSG     pMsg,
    void            *pPayload,
    LwU32            queueIdLogical,
    FlcnQMgrClientCallback pCallback,
    void            *pCallbackParams,
    LwU32           *pSeqDesc,
    LWSWITCH_TIMEOUT *pTimeout
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueCmdPostNonBlocking != (void *)0);
    return pFlcn->pHal->queueCmdPostNonBlocking(device, pFlcn, pCmd, pMsg, pPayload, queueIdLogical, pCallback, pCallbackParams, pSeqDesc, pTimeout);
}

LW_STATUS
flcnQueueCmdPostBlocking
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PRM_FLCN_CMD     pCmd,
    PRM_FLCN_MSG     pMsg,
    void            *pPayload,
    LwU32            queueIdLogical,
    LwU32           *pSeqDesc,
    LWSWITCH_TIMEOUT *pTimeout
)
{
    LW_STATUS status;

    status = flcnQueueCmdPostNonBlocking(device, pFlcn, pCmd, pMsg, pPayload,
                    queueIdLogical, NULL, NULL, pSeqDesc, pTimeout);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_COMMAND_QUEUE,
            "Failed to post command to SOE\n");
        return status;
    }

    status = flcnQueueCmdWait(device, pFlcn, *pSeqDesc, pTimeout);
    if (status == LW_ERR_TIMEOUT)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_SOE_TIMEOUT,
                "Timed out while waiting for SOE command completion\n");
        flcnQueueCmdCancel(device, pFlcn, *pSeqDesc);
    }

    return status;
}

LW_STATUS
flcnQueueCmdWait
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            seqDesc,
    LWSWITCH_TIMEOUT *pTimeout
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueCmdWait != (void *)0);

    return pFlcn->pHal->queueCmdWait(device, pFlcn, seqDesc, pTimeout);
}

LwU8
flcnCoreRevisionGet
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->coreRevisionGet != (void *)0);
    return pFlcn->pHal->coreRevisionGet(device, pFlcn);
}

void
flcnMarkNotReady
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->markNotReady != (void *)0);
    pFlcn->pHal->markNotReady(device, pFlcn);
}

LW_STATUS
flcnCmdQueueHeadGet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pHead
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->cmdQueueHeadGet != (void *)0);
    return pFlcn->pHal->cmdQueueHeadGet(device, pFlcn, pQueue, pHead);
}

LW_STATUS
flcnMsgQueueHeadGet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pHead
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->msgQueueHeadGet != (void *)0);
    return pFlcn->pHal->msgQueueHeadGet(device, pFlcn, pQueue, pHead);
}

LW_STATUS
flcnCmdQueueTailGet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pTail
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->cmdQueueTailGet != (void *)0);
    return pFlcn->pHal->cmdQueueTailGet(device, pFlcn, pQueue, pTail);
}

LW_STATUS
flcnMsgQueueTailGet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pTail
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->msgQueueTailGet != (void *)0);
    return pFlcn->pHal->msgQueueTailGet(device, pFlcn, pQueue, pTail);
}

LW_STATUS
flcnCmdQueueHeadSet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            head
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->cmdQueueHeadSet != (void *)0);
    return pFlcn->pHal->cmdQueueHeadSet(device, pFlcn, pQueue, head);
}

LW_STATUS
flcnMsgQueueHeadSet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            head
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->msgQueueHeadSet != (void *)0);
    return pFlcn->pHal->msgQueueHeadSet(device, pFlcn, pQueue, head);
}

LW_STATUS
flcnCmdQueueTailSet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            tail
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->cmdQueueTailSet != (void *)0);
    return pFlcn->pHal->cmdQueueTailSet(device, pFlcn, pQueue, tail);
}

LW_STATUS
flcnMsgQueueTailSet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            tail
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->msgQueueTailSet != (void *)0);
    return pFlcn->pHal->msgQueueTailSet(device, pFlcn, pQueue, tail);
}

PFLCN_QMGR_SEQ_INFO
flcnQueueSeqInfoFind
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            seqDesc
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoFind != (void *)0);
    return pFlcn->pHal->queueSeqInfoFind(device, pFlcn, seqDesc);
}

PFLCN_QMGR_SEQ_INFO
flcnQueueSeqInfoAcq
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoAcq != (void *)0);
    return pFlcn->pHal->queueSeqInfoAcq(device, pFlcn);
}

void
flcnQueueSeqInfoRel
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoRel != (void *)0);
    pFlcn->pHal->queueSeqInfoRel(device, pFlcn, pSeqInfo);
}

void
flcnQueueSeqInfoStateInit
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoStateInit != (void *)0);
    pFlcn->pHal->queueSeqInfoStateInit(device, pFlcn);
}

void
flcnQueueSeqInfoCancelAll
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoCancelAll != (void *)0);
    pFlcn->pHal->queueSeqInfoCancelAll(device, pFlcn);
}

LW_STATUS
flcnQueueSeqInfoFree
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueSeqInfoFree != (void *)0);
    return pFlcn->pHal->queueSeqInfoFree(device, pFlcn, pSeqInfo);
}

LW_STATUS
flcnQueueEventRegister
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            unitId,
    LwU8            *pMsg,
    FlcnQMgrClientCallback pCallback,
    void            *pParams,
    LwU32           *pEvtDesc
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueEventRegister != (void *)0);
    return pFlcn->pHal->queueEventRegister(device, pFlcn, unitId, pMsg, pCallback, pParams, pEvtDesc);
}

LW_STATUS
flcnQueueEventUnregister
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            evtDesc
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueEventUnregister != (void *)0);
    return pFlcn->pHal->queueEventUnregister(device, pFlcn, evtDesc);
}

LW_STATUS
flcnQueueEventHandle
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    RM_FLCN_MSG     *pMsg,
    LW_STATUS        evtStatus
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueEventHandle != (void *)0);
    return pFlcn->pHal->queueEventHandle(device, pFlcn, pMsg, evtStatus);
}

LW_STATUS
flcnQueueResponseHandle
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    RM_FLCN_MSG     *pMsg
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueResponseHandle != (void *)0);
    return pFlcn->pHal->queueResponseHandle(device, pFlcn, pMsg);
}

LwU32
flcnQueueCmdStatus
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            seqDesc
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->queueCmdStatus != (void *)0);
    return pFlcn->pHal->queueCmdStatus(device, pFlcn, seqDesc);
}

LW_STATUS
flcnDmemCopyFrom
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            src,
    LwU8            *pDst,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->dmemCopyFrom != (void *)0);
    return pFlcn->pHal->dmemCopyFrom(device, pFlcn, src, pDst, sizeBytes, port);
}

LW_STATUS
flcnDmemCopyTo
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->dmemCopyTo != (void *)0);
    return pFlcn->pHal->dmemCopyTo(device, pFlcn, dst, pSrc, sizeBytes, port);
}

void
flcnPostDiscoveryInit
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->postDiscoveryInit != (void *)0);
    pFlcn->pHal->postDiscoveryInit(device, pFlcn);
}

void
flcnDbgInfoDmemOffsetSet
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU16            debugInfoDmemOffset
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->dbgInfoDmemOffsetSet != (void *)0);
    pFlcn->pHal->dbgInfoDmemOffsetSet(device, pFlcn, debugInfoDmemOffset);
}



// HAL Interfaces
LW_STATUS
flcnConstruct_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->construct != (void *)0);
    return pFlcn->pHal->construct(device, pFlcn);
}

void
flcnDestruct_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->destruct != (void *)0);
    pFlcn->pHal->destruct(device, pFlcn);
}

LwU32
flcnRegRead_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn,
    LwU32                   offset
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->regRead != (void *)0);
    return pFlcn->pHal->regRead(device, pFlcn, offset);
}

void
flcnRegWrite_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn,
    LwU32                   offset,
    LwU32                   data
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->regWrite != (void *)0);
    pFlcn->pHal->regWrite(device, pFlcn, offset, data);
}

const char *
flcnGetName_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->getName != (void *)0);
    return pFlcn->pHal->getName(device, pFlcn);
}

LwU8
flcnReadCoreRev_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->readCoreRev != (void *)0);
    return pFlcn->pHal->readCoreRev(device, pFlcn);
}

void
flcnGetCoreInfo_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->getCoreInfo != (void *)0);
    pFlcn->pHal->getCoreInfo(device, pFlcn);
}

LW_STATUS
flcnDmemTransfer_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            src,
    LwU8            *pDst,
    LwU32            sizeBytes,
    LwU8             port,
    LwBool           bCopyFrom
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->dmemTransfer != (void *)0);
    return pFlcn->pHal->dmemTransfer(device, pFlcn, src, pDst, sizeBytes, port, bCopyFrom);
}

void
flcnIntrRetrigger_HAL
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->intrRetrigger != (void *)0);
    pFlcn->pHal->intrRetrigger(device, pFlcn);
}

LwBool
flcnAreEngDescsInitialized_HAL
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->areEngDescsInitialized != (void *)0);
    return pFlcn->pHal->areEngDescsInitialized(device, pFlcn);
}

LW_STATUS
flcnWaitForResetToFinish_HAL
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->waitForResetToFinish != (void *)0);
    return pFlcn->pHal->waitForResetToFinish(device, pFlcn);
}

void
flcnDbgInfoCapturePcTrace_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    if (pFlcn->pHal->dbgInfoCapturePcTrace == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pFlcn->pHal->dbgInfoCapturePcTrace(device, pFlcn);
}

void
flcnDbgInfoCaptureRiscvPcTrace_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    if (pFlcn->pHal->dbgInfoCaptureRiscvPcTrace == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pFlcn->pHal->dbgInfoCaptureRiscvPcTrace(device, pFlcn);
}


LwU32
flcnDmemSize_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    if (pFlcn->pHal->dmemSize == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pFlcn->pHal->dmemSize(device, pFlcn);    
}

LwU32
flcnSetImemAddr_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst
)
{
    if (pFlcn->pHal->setImemAddr == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pFlcn->pHal->setImemAddr(device, pFlcn, dst);    
}

void
flcnImemCopyTo_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwBool           bSelwre,
    LwU32            tag,
    LwU8             port
)
{
    if (pFlcn->pHal->imemCopyTo == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pFlcn->pHal->imemCopyTo(device, pFlcn, dst, pSrc, sizeBytes, bSelwre, tag, port);    
}

LwU32
flcnSetDmemAddr_HAL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst
)
{
    if (pFlcn->pHal->setDmemAddr == (void *)0)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pFlcn->pHal->setDmemAddr(device, pFlcn, dst);    
}

LwU32
flcnRiscvRegRead_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn,
    LwU32                   offset
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->riscvRegRead != (void *)0);
    return pFlcn->pHal->riscvRegRead(device, pFlcn, offset);
}

void
flcnRiscvRegWrite_HAL
(
    struct lwswitch_device *device,
    PFLCN                   pFlcn,
    LwU32                   offset,
    LwU32                   data
)
{
    LWSWITCH_ASSERT(pFlcn->pHal->riscvRegWrite != (void *)0);
    pFlcn->pHal->riscvRegWrite(device, pFlcn, offset, data);
}
