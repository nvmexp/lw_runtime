/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HALDEFS_FLCN_LWSWITCH_H_
#define _HALDEFS_FLCN_LWSWITCH_H_

#include "g_lwconfig.h"
#include "lwstatus.h"
#include "flcn/flcnqueue_lwswitch.h"
#include "flcnifcmn.h"

struct lwswitch_device;
struct LWSWITCH_TIMEOUT;
struct FLCN;
union  RM_FLCN_MSG;
union  RM_FLCN_CMD;
struct FLCNQUEUE;
struct FLCN_QMGR_SEQ_INFO;

typedef struct {
    // OBJECT Interfaces
    LW_STATUS   (*queueReadData)                    (struct lwswitch_device *, struct FLCN *, LwU32 queueId, void *pData, LwBool bMsg);
    LW_STATUS   (*queueCmdWrite)                    (struct lwswitch_device *, struct FLCN *, LwU32 queueId, union RM_FLCN_CMD *pCmd, struct LWSWITCH_TIMEOUT *pTimeout);
    LW_STATUS   (*queueCmdCancel)                   (struct lwswitch_device *, struct FLCN *, LwU32 seqDesc);
    LW_STATUS   (*queueCmdPostNonBlocking)          (struct lwswitch_device *, struct FLCN *, union RM_FLCN_CMD *pCmd, union RM_FLCN_MSG *pMsg, void *pPayload, LwU32 queueIdLogical, FlcnQMgrClientCallback pCallback, void *pCallbackParams, LwU32 *pSeqDesc, struct LWSWITCH_TIMEOUT *pTimeout);
    LW_STATUS   (*queueCmdWait)                     (struct lwswitch_device *, struct FLCN *, LwU32 seqDesc, struct LWSWITCH_TIMEOUT *pTimeout);
    LwU8        (*coreRevisionGet)                  (struct lwswitch_device *, struct FLCN *);
    void        (*markNotReady)                     (struct lwswitch_device *, struct FLCN *);
    LW_STATUS   (*cmdQueueHeadGet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 *pHead);
    LW_STATUS   (*msgQueueHeadGet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 *pHead);
    LW_STATUS   (*cmdQueueTailGet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 *pTail);
    LW_STATUS   (*msgQueueTailGet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 *pTail);
    LW_STATUS   (*cmdQueueHeadSet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 head);
    LW_STATUS   (*msgQueueHeadSet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 head);
    LW_STATUS   (*cmdQueueTailSet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 tail);
    LW_STATUS   (*msgQueueTailSet)                  (struct lwswitch_device *, struct FLCN *, struct FLCNQUEUE *pQueue, LwU32 tail);
    struct FLCN_QMGR_SEQ_INFO *(*queueSeqInfoFind)  (struct lwswitch_device *, struct FLCN *, LwU32 seqDesc);
    struct FLCN_QMGR_SEQ_INFO *(*queueSeqInfoAcq)   (struct lwswitch_device *, struct FLCN *);
    void        (*queueSeqInfoRel)                  (struct lwswitch_device *, struct FLCN *, struct FLCN_QMGR_SEQ_INFO *pSeqInfo);
    void        (*queueSeqInfoStateInit)            (struct lwswitch_device *, struct FLCN *);
    void        (*queueSeqInfoCancelAll)            (struct lwswitch_device *, struct FLCN *);
    LW_STATUS   (*queueSeqInfoFree)                 (struct lwswitch_device *, struct FLCN *, struct FLCN_QMGR_SEQ_INFO *);
    LW_STATUS   (*queueEventRegister)               (struct lwswitch_device *, struct FLCN *, LwU32 unitId, LwU8 *pMsg, FlcnQMgrClientCallback pCallback, void *pParams, LwU32 *pEvtDesc);
    LW_STATUS   (*queueEventUnregister)             (struct lwswitch_device *, struct FLCN *, LwU32 evtDesc);
    LW_STATUS   (*queueEventHandle)                 (struct lwswitch_device *, struct FLCN *, union RM_FLCN_MSG *pMsg, LW_STATUS evtStatus);
    LW_STATUS   (*queueResponseHandle)              (struct lwswitch_device *, struct FLCN *, union RM_FLCN_MSG *pMsg);
    LwU32       (*queueCmdStatus)                   (struct lwswitch_device *, struct FLCN *, LwU32 seqDesc);
    LW_STATUS   (*dmemCopyFrom)                     (struct lwswitch_device *, struct FLCN *, LwU32 src, LwU8 *pDst, LwU32 sizeBytes, LwU8 port);
    LW_STATUS   (*dmemCopyTo)                       (struct lwswitch_device *, struct FLCN *, LwU32 dst, LwU8 *pSrc, LwU32 sizeBytes, LwU8 port);
    void        (*postDiscoveryInit)                (struct lwswitch_device *, struct FLCN *);
    void        (*dbgInfoDmemOffsetSet)             (struct lwswitch_device *, struct FLCN *, LwU16 debugInfoDmemOffset);


    //HAL Interfaces
    LW_STATUS   (*construct)                        (struct lwswitch_device *, struct FLCN *);
    void        (*destruct)                         (struct lwswitch_device *, struct FLCN *);
    LwU32       (*regRead)                          (struct lwswitch_device *, struct FLCN *, LwU32 offset);
    void        (*regWrite)                         (struct lwswitch_device *, struct FLCN *, LwU32 offset, LwU32 data);
    const char *(*getName)                          (struct lwswitch_device *, struct FLCN *);
    LwU8        (*readCoreRev)                      (struct lwswitch_device *, struct FLCN *);
    void        (*getCoreInfo)                      (struct lwswitch_device *, struct FLCN *);
    LW_STATUS   (*dmemTransfer)                     (struct lwswitch_device *, struct FLCN *, LwU32 src, LwU8 *pDst, LwU32 sizeBytes, LwU8 port, LwBool bCopyFrom);
    void        (*intrRetrigger)                    (struct lwswitch_device *, struct FLCN *);
    LwBool      (*areEngDescsInitialized)           (struct lwswitch_device *, struct FLCN *);
    LW_STATUS   (*waitForResetToFinish)             (struct lwswitch_device *, struct FLCN *);
    void        (*dbgInfoCapturePcTrace)            (struct lwswitch_device *, struct FLCN *);
    void        (*dbgInfoCaptureRiscvPcTrace)       (struct lwswitch_device *, struct FLCN *);
    LwU32       (*dmemSize)                         (struct lwswitch_device *, struct FLCN *);
    LwU32       (*setImemAddr)                      (struct lwswitch_device *, struct FLCN *, LwU32 dst);
    void        (*imemCopyTo)                       (struct lwswitch_device *, struct FLCN *, LwU32 dst, LwU8 *pSrc, LwU32 sizeBytes, LwBool bSelwre, LwU32 tag, LwU8 port);
    LwU32       (*setDmemAddr)                      (struct lwswitch_device *, struct FLCN *, LwU32 dst);
    LwU32       (*riscvRegRead)                     (struct lwswitch_device *, struct FLCN *, LwU32 offset);
    void        (*riscvRegWrite)                    (struct lwswitch_device *, struct FLCN *, LwU32 offset, LwU32 data);
} flcn_hal;

void flcnQueueSetupHal(struct FLCN *pFlcn);
void flcnRtosSetupHal(struct FLCN *pFlcn);
void flcnQueueRdSetupHal(struct FLCN *pFlcn);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
void flcnSetupHal_LS10(struct FLCN *pFlcn);
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void flcnSetupHal_LR10(struct FLCN *pFlcn);
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
void flcnSetupHal_SV10(struct FLCN *pFlcn);
#endif

void flcnSetupHal_v03_00(struct FLCN *pFlcn);
void flcnSetupHal_v04_00(struct FLCN *pFlcn);
void flcnSetupHal_v05_01(struct FLCN *pFlcn);
void flcnSetupHal_v06_00(struct FLCN *pFlcn);

#endif //_HALDEFS_FLCN_LWSWITCH_H_
