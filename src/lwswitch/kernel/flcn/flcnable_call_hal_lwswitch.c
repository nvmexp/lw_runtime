/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "flcn/haldefs_flcnable_lwswitch.h"
#include "flcn/flcnable_lwswitch.h"

#include "flcnifcmn.h"

#include "export_lwswitch.h"
#include "common_lwswitch.h"

typedef struct FALCON_EXTERNAL_CONFIG FALCON_EXTERNAL_CONFIG, *PFALCON_EXTERNAL_CONFIG;
typedef struct FLCN_QMGR_SEQ_INFO FLCN_QMGR_SEQ_INFO, *PFLCN_QMGR_SEQ_INFO;
typedef union  RM_FLCN_CMD RM_FLCN_CMD, *PRM_FLCN_CMD;
typedef union  RM_FLCN_MSG RM_FLCN_MSG, *PRM_FLCN_MSG;
typedef struct ENGINE_DESCRIPTOR_TYPE ENGINE_DESCRIPTOR_TYPE, *PENGINE_DESCRIPTOR_TYPE;


// OBJECT Interfaces
LwU8
flcnableReadCoreRev
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->readCoreRev != (void *)0);
    return pFlcnable->pHal->readCoreRev(device, pFlcnable);
}

void
flcnableGetExternalConfig
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    PFALCON_EXTERNAL_CONFIG pConfig
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->getExternalConfig != (void *)0);
    pFlcnable->pHal->getExternalConfig(device, pFlcnable, pConfig);
}

void
flcnableEmemCopyFrom
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            src,
    LwU8            *pDst,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->ememCopyFrom != (void *)0);
    pFlcnable->pHal->ememCopyFrom(device, pFlcnable, src, pDst, sizeBytes, port);
}

void
flcnableEmemCopyTo
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->ememCopyTo != (void *)0);
    pFlcnable->pHal->ememCopyTo(device, pFlcnable, dst, pSrc, sizeBytes, port);
}

LW_STATUS
flcnableHandleInitEvent
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    RM_FLCN_MSG     *pGenMsg
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->handleInitEvent != (void *)0);
    return pFlcnable->pHal->handleInitEvent(device, pFlcnable, pGenMsg);
}

PFLCN_QMGR_SEQ_INFO
flcnableQueueSeqInfoGet
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            seqIndex
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->queueSeqInfoGet != (void *)0);
    return pFlcnable->pHal->queueSeqInfoGet(device, pFlcnable, seqIndex);
}

void
flcnableQueueSeqInfoClear
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->queueSeqInfoClear != (void *)0);
    pFlcnable->pHal->queueSeqInfoClear(device, pFlcnable, pSeqInfo);
}

void
flcnableQueueSeqInfoFree
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->queueSeqInfoFree != (void *)0);
    pFlcnable->pHal->queueSeqInfoFree(device, pFlcnable, pSeqInfo);
}

LwBool
flcnableQueueCmdValidate
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    RM_FLCN_CMD     *pCmd,
    RM_FLCN_MSG     *pMsg,
    void            *pPayload,
    LwU32            queueIdLogical
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->queueCmdValidate != (void *)0);
    return pFlcnable->pHal->queueCmdValidate(device, pFlcnable, pCmd, pMsg, pPayload, queueIdLogical);
}

LW_STATUS
flcnableQueueCmdPostExtension
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    RM_FLCN_CMD        *pCmd,
    RM_FLCN_MSG        *pMsg,
    void               *pPayload,
    LWSWITCH_TIMEOUT   *pTimeout,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->queueCmdPostExtension != (void *)0);
    return pFlcnable->pHal->queueCmdPostExtension(device, pFlcnable, pCmd, pMsg, pPayload, pTimeout, pSeqInfo);
}

void
flcnablePostDiscoveryInit
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->postDiscoveryInit != (void *)0);
    pFlcnable->pHal->postDiscoveryInit(device, pFlcnable);
}



// HAL Interfaces
LW_STATUS
flcnableConstruct_HAL
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->construct != (void *)0);
    return pFlcnable->pHal->construct(device, pFlcnable);
}

void
flcnableDestruct_HAL
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->destruct != (void *)0);
    pFlcnable->pHal->destruct(device, pFlcnable);
}

void
flcnableFetchEngines_HAL
(
    lwswitch_device *device,
    FLCNABLE        *pFlcnable,
    ENGINE_DESCRIPTOR_TYPE  *pEngDeslwc,
    ENGINE_DESCRIPTOR_TYPE  *pEngDescBc
)
{
    LWSWITCH_ASSERT(pFlcnable->pHal->fetchEngines != (void *)0);
    pFlcnable->pHal->fetchEngines(device, pFlcnable, pEngDeslwc, pEngDescBc);
}
