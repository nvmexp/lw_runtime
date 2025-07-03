/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "soe/soe_lwswitch.h"
#include "soe/soe_priv_lwswitch.h"

#include "flcn/haldefs_flcnable_lwswitch.h"
#include "flcn/haldefs_flcn_lwswitch.h"
#include "flcn/flcn_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"
#include "common_lwswitch.h"

static LW_STATUS _soeGetInitMessage(lwswitch_device *device, PSOE pSoe, RM_FLCN_MSG_SOE *pMsg);

/*!
 * Use the SOE INIT Message to construct and initialize all SOE Queues.
 *
 * @param[in]      device lwswitch_device pointer
 * @param[in]      pSoe   SOE object pointer
 * @param[in]      pMsg   Pointer to the INIT Message
 *
 * @return 'LW_OK' upon successful creation of all SOE Queues
 */
static LW_STATUS
_soeQMgrCreateQueuesFromInitMsg
(
    lwswitch_device  *device,
    PFLCNABLE         pSoe,
    RM_FLCN_MSG_SOE  *pMsg
)
{
    RM_SOE_INIT_MSG_SOE_INIT *pInit;
    LwU32                     i;
    LwU32                     queueLogId;
    LW_STATUS                 status;
    FLCNQUEUE                *pQueue;
    PFLCN                     pFlcn = ENG_GET_FLCN(pSoe);
    PFALCON_QUEUE_INFO        pQueueInfo;

    LWSWITCH_ASSERT(pFlcn != NULL);

    pQueueInfo = pFlcn->pQueueInfo;
    LWSWITCH_ASSERT(pQueueInfo != NULL);

    pInit = &pMsg->msg.init.soeInit;
    LWSWITCH_ASSERT(pInit->numQueues <= pFlcn->numQueues);

    for (i = 0; i < pFlcn->numQueues; i++)
    {
        queueLogId = pInit->qInfo[i].queueLogId;
        LWSWITCH_ASSERT(queueLogId < pFlcn->numQueues);
        pQueue = &pQueueInfo->pQueues[queueLogId];
        status = flcnQueueConstruct_dmem_lwswitch(
                     device,
                     pFlcn,
                     &pQueue,                                  // ppQueue
                     queueLogId,                               // Logical ID of the queue
                     pInit->qInfo[i].queuePhyId,               // Physical ID of the queue
                     pInit->qInfo[i].queueOffset,              // offset
                     pInit->qInfo[i].queueSize,                // size
                     RM_FLCN_QUEUE_HDR_SIZE);                  // cmdHdrSize
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                        "%s: Error constructing SOE Queue (status="
                        "0x%08x).\n", __FUNCTION__, status);
            LWSWITCH_ASSERT(0);
            return status;
        }
    }
    return LW_OK;
}

/*!
 * Purges all the messages from the SOE's message queue.  Each message will
 * be analyzed, clients will be notified of status, and events will be routed
 * to all registered event listeners.
 *
 * @param[in]  device lwswitch_device pointer
 * @param[in]  pSoe   SOE object pointer
 *
 * @return 'LW_OK' if the message queue was successfully purged.
 */
static LW_STATUS
_soeProcessMessages_IMPL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    RM_FLCN_MSG_SOE  soeMessage;
    LW_STATUS        status;
    PFLCN            pFlcn  = ENG_GET_FLCN(pSoe);

    // keep processing messages until no more exist in the message queue
    while (LW_OK == (status = flcnQueueReadData(
                                     device,
                                     pFlcn,
                                     SOE_RM_MSGQ_LOG_ID,
                                     (RM_FLCN_MSG *)&soeMessage, LW_TRUE)))
    {
        LWSWITCH_PRINT(device, INFO,
                    "%s: unitId=0x%02x, size=0x%02x, ctrlFlags=0x%02x, " \
                    "seqNumId=0x%02x\n",
                    __FUNCTION__,
                    soeMessage.hdr.unitId,
                    soeMessage.hdr.size,
                    soeMessage.hdr.ctrlFlags,
                    soeMessage.hdr.seqNumId);

        // check to see if the message is a reply or an event.
        if ((soeMessage.hdr.ctrlFlags &= RM_FLCN_QUEUE_HDR_FLAGS_EVENT) != 0)
        {
            flcnQueueEventHandle(device, pFlcn, (RM_FLCN_MSG *)&soeMessage, LW_OK);
        }
        // the message is a response from a previously queued command
        else
        {
            flcnQueueResponseHandle(device, pFlcn, (RM_FLCN_MSG *)&soeMessage);
        }
    }

    //
    // Status LW_ERR_NOT_READY implies, Queue is empty.
    // Log the message in other error cases.
    //
    if (status != LW_ERR_NOT_READY)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: unexpected error while purging message queue (status=0x%x).\n",
            __FUNCTION__, (status));
    }

    return status;
}

/*!
 * This function exists to solve a natural chicken-and-egg problem that arises
 * due to the fact that queue information (location, size, id, etc...) is
 * relayed to the RM as a message in a queue.  Queue construction is done when
 * the message arives and the normal queue read/write functions are not
 * available until construction is complete.  Construction cannot be done until
 * the message is read from the queue.  Therefore, the very first message read
 * from the Message Queue must be considered as a special-case and must NOT use
 * any functionality provided by the SOE's queue manager.
 *
 * @param[in]  device  lwswitch_device pointer
 * @param[in]  pSoe    SOE object pointer
 *
 * @return 'LW_OK'
 *     Upon successful extraction and processing of the first SOE message.
 */
static LW_STATUS
_soeProcessMessagesPreInit_IMPL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    RM_FLCN_MSG_SOE   msg;
    LW_STATUS        status;
    PFLCN            pFlcn  = ENG_GET_FLCN(pSoe);

    // extract the "INIT" message (this is never expected to fail)
    status = _soeGetInitMessage(device, pSoe, &msg);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "%s: Failed to extract the INIT message "
                    "from the SOE Message Queue (status=0x%08x).",
                    __FUNCTION__, status);
        LWSWITCH_ASSERT(0);
        return status;
    }

    //
    // Now hookup the "real" message-processing function and handle the "INIT"
    // message.
    //
    pSoe->base.pHal->processMessages = _soeProcessMessages_IMPL;
    return flcnQueueEventHandle(device, pFlcn, (RM_FLCN_MSG *)&msg, LW_OK);
}

/*!
 * @brief Process the "INIT" message sent from the SOE ucode application.
 *
 * When the SOE ucode is done initializing, it will post an INIT message in
 * the Message Queue that contains all the necessary attributes that are
 * needed to enqueuing commands and extracting messages from the queues.
 * The packet will also contain the offset and size of portion of DMEM that
 * the RM must manage.  Upon receiving this message it will be assume that
 * the SOE is ready to start accepting commands.
 *
 * @param[in]  device  lwswitch_device pointer
 * @param[in]  pSoe    SOE object pointer
 * @param[in]  pMsg    Pointer to the event's message data
 *
 * @return 'LW_OK' if the event was successfully handled.
 */
static LW_STATUS
_soeHandleInitEvent_IMPL
(
    lwswitch_device  *device,
    PFLCNABLE         pSoe,
    RM_FLCN_MSG      *pGenMsg
)
{
    LW_STATUS         status;
    PFLCN             pFlcn = ENG_GET_FLCN(pSoe);
    RM_FLCN_MSG_SOE *pMsg  = (RM_FLCN_MSG_SOE *)pGenMsg;

    if (pFlcn == NULL)
    {
        LWSWITCH_ASSERT(pFlcn != NULL);
        return LW_ERR_ILWALID_POINTER;
    }

    LWSWITCH_PRINT(device, INFO,
                "%s: Received INIT message from SOE\n",
                __FUNCTION__);

    //
    // Pass the INIT message to the queue manager to allow it to create the
    // queues.
    //
    status = _soeQMgrCreateQueuesFromInitMsg(device, pSoe, pMsg);
    if (status != LW_OK)
    {
        LWSWITCH_ASSERT(0);
        return status;
    }

    flcnDbgInfoDmemOffsetSet(device, pFlcn,
        pMsg->msg.init.soeInit.osDebugEntryPoint);

    // the SOE ucode is now initialized and ready to accept commands
    pFlcn->bOSReady = LW_TRUE;

    return LW_OK;
}

/*!
 * @brief Read the INIT message directly out of the Message Queue.
 *
 * This function accesses the Message Queue directly using the HAL.  It does
 * NOT and may NOT use the queue manager as it has not yet been constructed and
 * initialized.  The Message Queue may not be empty when this function is called
 * and the first message in the queue MUST be the INIT message.
 *
 * @param[in]   device  lwswitch_device pointer
 * @param[in]   pSoe    SOE object pointer
 * @param[out]  pMsg    Message structure to fill with the INIT message data
 *
 * @return 'LW_OK' upon successful extraction of the INIT message.
 * @return
 *     'LW_ERR_ILWALID_STATE' if the first message found was not an INIT
 *     message or if the message was improperly formatted.
 */
static LW_STATUS
_soeGetInitMessage
(
    lwswitch_device  *device,
    PSOE              pSoe,
    RM_FLCN_MSG_SOE  *pMsg
)
{
    PFLCN               pFlcn   = ENG_GET_FLCN(pSoe);
    LW_STATUS           status  = LW_OK;
    LwU32               tail    = 0;
    PFALCON_QUEUE_INFO  pQueueInfo;
    // on the GPU, rmEmemPortId = sec2RmEmemPortIdGet_HAL(...);
    LwU8                rmEmemPortId = 0;

    if (pFlcn == NULL)
    {
        LWSWITCH_ASSERT(pFlcn != NULL);
        return LW_ERR_ILWALID_POINTER;
    }

    pQueueInfo = pFlcn->pQueueInfo;
    if (pQueueInfo == NULL)
    {
        LWSWITCH_ASSERT(pQueueInfo != NULL);
        return LW_ERR_ILWALID_POINTER;
    }

    //
    // Message queue 0 is used by SOE to communicate with RM
    // Check SOE_CMDMGMT_MSG_QUEUE_RM in //uproc/soe/inc/soe_cmdmgmt.h
    //
    pQueueInfo->pQueues[SOE_RM_MSGQ_LOG_ID].queuePhyId = 0;

    // read the header starting at the current tail position
    (void)flcnMsgQueueTailGet(device, pFlcn,
        &pQueueInfo->pQueues[SOE_RM_MSGQ_LOG_ID], &tail);
    if (pFlcn->bEmemEnabled)
    {
        //
        // We use the offset in DMEM for the src address, since
        // EmemCopyFrom automatically colwerts it to the offset in EMEM
        //
        flcnableEmemCopyFrom(
            device, pFlcn->pFlcnable,
            tail,                   // src
            (LwU8 *)&pMsg->hdr,     // pDst
            RM_FLCN_QUEUE_HDR_SIZE, // numBytes
            rmEmemPortId);          // port
    }
    else
    {
        status = flcnDmemCopyFrom(device,
                                  pFlcn,
                                  tail,                     // src
                                  (LwU8 *)&pMsg->hdr,       // pDst
                                  RM_FLCN_QUEUE_HDR_SIZE,   // numBytes
                                  0);                       // port
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to copy from SOE DMEM\n", __FUNCTION__);
            LWSWITCH_ASSERT(0);
            goto _soeGetInitMessage_exit;
        }
    }

    if (pMsg->hdr.unitId != RM_SOE_UNIT_INIT)
    {
        status = LW_ERR_ILWALID_STATE;
        LWSWITCH_ASSERT(0);
        goto _soeGetInitMessage_exit;
    }

    // read the message body and update the tail position
    if (pFlcn->bEmemEnabled)
    {
        //
        // We use the offset in DMEM for the src address, since
        // EmemCopyFrom automatically colwerts it to the offset in EMEM
        //
        flcnableEmemCopyFrom(
            device, pFlcn->pFlcnable,
            tail + RM_FLCN_QUEUE_HDR_SIZE,              // src
            (LwU8 *)&pMsg->msg,                         // pDst
            pMsg->hdr.size - RM_FLCN_QUEUE_HDR_SIZE,    // numBytes
            rmEmemPortId);                              // port
    }
    else
    {
        status = flcnDmemCopyFrom(device,
            pFlcn,
            tail + RM_FLCN_QUEUE_HDR_SIZE,              // src
            (LwU8 *)&pMsg->msg,                         // pDst
            pMsg->hdr.size - RM_FLCN_QUEUE_HDR_SIZE,    // numBytes
            0);                                         // port
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to copy from SOE DMEM\n", __FUNCTION__);
            LWSWITCH_ASSERT(0);
            goto _soeGetInitMessage_exit;
        }
    }

    tail += LW_ALIGN_UP(pMsg->hdr.size, SOE_DMEM_ALIGNMENT);
    flcnMsgQueueTailSet(device, pFlcn,
        &pQueueInfo->pQueues[SOE_RM_MSGQ_LOG_ID], tail);

_soeGetInitMessage_exit:
    return status;
}

/*!
 * Copies 'sizeBytes' from DMEM address 'src' to 'pDst' using EMEM access port.
 *
 * The address must be located in the EMEM region located directly above the
 * maximum virtual address of DMEM.
 *
 * @param[in]   device      lwswitch_device pointer
 * @param[in]   pSoe        SOE pointer
 * @param[in]   src         The DMEM address for the source of the copy
 * @param[out]  pDst        Pointer to write with copied data from EMEM
 * @param[in]   sizeBytes   The number of bytes to copy from EMEM
 * @param[in]   port        EMEM port
 */
static void
_soeEmemCopyFrom_IMPL
(
    lwswitch_device    *device,
    FLCNABLE           *pSoe,
    LwU32               src,
    LwU8               *pDst,
    LwU32               sizeBytes,
    LwU8                port
)
{
    soeEmemTransfer_HAL(device, (PSOE)pSoe, src, pDst, sizeBytes, port, LW_TRUE);
}

/*!
 * Copies 'sizeBytes' from 'pDst' to DMEM address 'dst' using EMEM access port.
 *
 * The address must be located in the EMEM region located directly above the
 * maximum virtual address of DMEM.
 *
 * @param[in]  device      lwswitch_device pointer
 * @param[in]  pSoe        SOE pointer
 * @param[in]  dst         The DMEM address for the copy destination.
 * @param[in]  pSrc        The pointer to the buffer containing the data to copy
 * @param[in]  sizeBytes   The number of bytes to copy into EMEM
 * @param[in]  port        EMEM port
 */
static void
_soeEmemCopyTo_IMPL
(
    lwswitch_device    *device,
    FLCNABLE           *pSoe,
    LwU32               dst,
    LwU8               *pSrc,
    LwU32               sizeBytes,
    LwU8                port
)
{
    soeEmemTransfer_HAL(device, (PSOE)pSoe, dst, pSrc, sizeBytes, port, LW_FALSE);
}

/*!
 * Loop until SOE RTOS is loaded and gives us an INIT message
 *
 * @param[in]  device  lwswitch_device object pointer
 * @param[in]  pSoe    SOE object pointer
 */
static LW_STATUS
_soeWaitForInitAck_IMPL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    PFLCN            pFlcn  = ENG_GET_FLCN(pSoe);
    // POBJMC           pMc    = GPU_GET_MC(device);
    LWSWITCH_TIMEOUT timeout;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS * 5, &timeout);
    while (!pFlcn->bOSReady && !lwswitch_timeout_check(&timeout))
    {
        // Once interrupt handling is ready, might need to replace this with
        //mcServiceSingle_HAL(device, pMc, MC_ENGINE_IDX_SOE, LW_FALSE);
        soeService_HAL(device, pSoe);
        lwswitch_os_sleep(1);
    }

    if (!pFlcn->bOSReady)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s Timeout while waiting for SOE bootup\n",
            __FUNCTION__);
        LWSWITCH_ASSERT(0);
        return LW_ERR_TIMEOUT;
    }
    return LW_OK;
}

/*!
 * @brief   Retrieves a pointer to the engine specific SEQ_INFO structure.
 *
 * @param[in]   device      lwswitch_device pointer
 * @param[in]   pSoe        SOE pointer
 * @param[in]   seqIndex    Index of the structure to retrieve
 *
 * @return  Pointer to the SEQ_INFO structure or NULL on invalid index.
 */
static PFLCN_QMGR_SEQ_INFO
_soeQueueSeqInfoGet_IMPL
(
    lwswitch_device    *device,
    FLCNABLE           *pSoe,
    LwU32               seqIndex
)
{
    FLCN *pFlcn = ENG_GET_FLCN(pSoe);

    if (seqIndex < pFlcn->numSequences)
    {
        return &(((PSOE)pSoe)->seqInfo[seqIndex]);
    }
    return NULL;
}

/*!
 * @copydoc flcnableQueueCmdValidate_IMPL
 */
static LwBool
_soeQueueCmdValidate_IMPL
(
    lwswitch_device    *device,
    FLCNABLE           *pSoe,
    PRM_FLCN_CMD        pCmd,
    PRM_FLCN_MSG        pMsg,
    void               *pPayload,
    LwU32               queueIdLogical
)
{
    PFLCN       pFlcn   = ENG_GET_FLCN(pSoe);
    FLCNQUEUE  *pQueue  = &pFlcn->pQueueInfo->pQueues[queueIdLogical];
    LwU32       cmdSize = pCmd->cmdGen.hdr.size;

    // Verify that the target queue ID represents a valid RM queue.
    if (queueIdLogical != SOE_RM_CMDQ_LOG_ID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid SOE command queue ID = 0x%x\n",
            __FUNCTION__, queueIdLogical);
        return LW_FALSE;
    }

    //
    // Command size cannot be larger than queue size / 2. Otherwise, it is
    // impossible to send two commands back to back if we start from the
    // beginning of the queue.
    //
    if (cmdSize > (pQueue->queueSize / 2))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid command (illegal size = 0x%x)\n",
            __FUNCTION__, cmdSize);
        return LW_FALSE;
    }

    // Validate the command's unit identifier.
    if (!RM_SOE_UNITID_IS_VALID(pCmd->cmdGen.hdr.unitId))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid unitID = %d\n",
            __FUNCTION__, pCmd->cmdGen.hdr.unitId);
        return LW_FALSE;
    }

    return LW_TRUE;
}

/* -------------------- Object construction/initialization ------------------- */

static void
soeSetupHal
(
    SOE      *pSoe,
    LwU32     pci_device_id
)
{
    soe_hal *pHal = NULL;
    flcnable_hal *pParentHal = NULL;

    if (lwswitch_is_lr10_device_id(pci_device_id))
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        soeSetupHal_LR10(pSoe);
#endif
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    else if (lwswitch_is_ls10_device_id(pci_device_id))
    {
        soeSetupHal_LS10(pSoe);
    }
#endif
    else
    {
        // we're on a device which doesn't support SOE
        LWSWITCH_PRINT(NULL, ERROR, "Tried to initialize SOE on device with no SOE\n");
        LWSWITCH_ASSERT(0);
    }

    pHal = pSoe->base.pHal;
    pParentHal = (flcnable_hal *)pHal;
    //set any functions we want to override
    pParentHal->handleInitEvent     = _soeHandleInitEvent_IMPL;
    pParentHal->ememCopyTo          = _soeEmemCopyTo_IMPL;
    pParentHal->ememCopyFrom        = _soeEmemCopyFrom_IMPL;
    pParentHal->queueSeqInfoGet     = _soeQueueSeqInfoGet_IMPL;
    pParentHal->queueCmdValidate    = _soeQueueCmdValidate_IMPL;

    //set any functions specific to SOE
    pHal->processMessages           = _soeProcessMessagesPreInit_IMPL;
    pHal->waitForInitAck            = _soeWaitForInitAck_IMPL;
}

SOE *
soeAllocNew(void)
{
    SOE *pSoe = lwswitch_os_malloc(sizeof(*pSoe));
    if (pSoe != NULL)
    {
        lwswitch_os_memset(pSoe, 0, sizeof(*pSoe));
    }

    return pSoe;
}

LwlStatus
soeInit
(
    lwswitch_device    *device,
    SOE                *pSoe,
    LwU32               pci_device_id
)
{
    LwlStatus retval;

    // allocate hal if a child class hasn't already
    if (pSoe->base.pHal == NULL)
    {
        soe_hal *pHal = pSoe->base.pHal = lwswitch_os_malloc(sizeof(*pHal));
        if (pHal == NULL)
        {
            LWSWITCH_PRINT(device, ERROR, "Flcn allocation failed!\n");
            retval = -LWL_NO_MEM;
            goto soe_init_fail;
        }
        lwswitch_os_memset(pHal, 0, sizeof(*pHal));
    }

    // init parent class
    retval = flcnableInit(device, (PFLCNABLE)pSoe, pci_device_id);
    if (retval != LWL_SUCCESS)
    {
        goto soe_init_fail;
    }

    soeSetupHal(pSoe, pci_device_id);

    return retval;
soe_init_fail:
    soeDestroy(device, pSoe);
    return retval;
}

// reverse of soeInit()
void
soeDestroy
(
    lwswitch_device    *device,
    SOE                *pSoe
)
{
    // destroy parent class
    flcnableDestroy(device, (PFLCNABLE)pSoe);

    if (pSoe->base.pHal != NULL)
    {
        lwswitch_os_free(pSoe->base.pHal);
        pSoe->base.pHal = NULL;
    }
}
