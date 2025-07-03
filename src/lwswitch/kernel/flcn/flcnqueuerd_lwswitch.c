/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"

#include "flcn/flcn_lwswitch.h"
#include "flcn/flcnqueue_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"

/*!
 * @file   flcnqueuerd_lwswitch.c
 */

static LW_STATUS _flcnQueueReaderGetNextHeader(lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, void *pData, LwBool bMsg);
static LW_STATUS _flcnQueueReaderReadHeader   (lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, void *pData, LwBool bMsg);
static LW_STATUS _flcnQueueReaderReadBody     (lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, void *pData, LwBool bMsg);

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_CLOSING(id, status)                    \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: error while closing queue (id=0x%x, status="        \
                 "0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_ILWALID_UNITID(id, unitId)                 \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: invalid unit-id read from queue (qid=0x%x, "        \
                 "uid=0x%x).\n", __FUNCTION__, (id), (unitId))

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_OPENING(id, status)                    \
             LWSWITCH_PRINT(device, WARN,                                       \
                 "%s: error while opening queue (id=0x%x, status="        \
                  "0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_READING(id, status)                    \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: error while reading from queue (id=0x%x, "          \
                 "status=0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGBODY(id, status)            \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: error reading body from queue (id=0x%x, "           \
                 "status=0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGHDR(id, status)             \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: error reading header from queue (id="               \
                 "0x%x, status=0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_ERR_READING_UNKNOWN_DATA(id, status)       \
             LWSWITCH_PRINT(device, ERROR,                                      \
                 "%s: unrecognizable data read from queue (id=0x%x, "     \
                 "status=0x%x).\n", __FUNCTION__, (id), (status))

#define  LWSWITCH_PRINT_QUEUE_READER_PRINT_HDR_READ_INFO(offset)                \
             LWSWITCH_PRINT(device, INFO,                                       \
                 "%s: Reading a header from DMEM @ 0x%x.\n",              \
                 __FUNCTION__, (offset))


/*!
 * Reads the queue and retrieves the next unread message/command.
 *
 * @param[in]       device       lwswitch device pointer
 * @param[in]       pFlcn        Falcon object pointer
 * @param[in]       queueLogId   Logical ID of the queue
 * @param[in,out]   pData        The buffer to fill with the queue data
 * @param[in]       bMsg         Message/Command
 *
 * @return  LW_OK            when the read operation is successful.
 *          LW_ERR_NOT_READY Queue is Empty
 *          LW_ERR_GENERIC         otherwise.
 */
static LW_STATUS
_flcnQueueReadData_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            queueLogId,
    void            *pData,
    LwBool           bMsg
)
{
    LW_STATUS           status    = LW_OK;
    LW_STATUS           retStatus = LW_OK;
    FLCNQUEUE          *pQueue;
    PFALCON_QUEUE_INFO  pQueueInfo;
    RM_FLCN_QUEUE_HDR   bufferGenHdr;

    LWSWITCH_ASSERT(pFlcn != NULL);

    pQueueInfo = pFlcn->pQueueInfo;
    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueueInfo->pQueues != NULL);

    pQueue = &pQueueInfo->pQueues[queueLogId];

    //
    // If the queue is empty, simply return LW_ERR_NOT_READY to indicate that a message is
    // not available.
    //
    if (pQueue->isEmpty(device, pFlcn, pQueue))
    {
        return LW_ERR_NOT_READY;
    }

    status = pQueue->openRead(device, pFlcn, pQueue);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_OPENING(pQueue->queueLogId, status);
        return status;
    }

    status = _flcnQueueReaderGetNextHeader(device, pFlcn, pQueue, pData, bMsg);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGHDR(pQueue->queueLogId, status);
        retStatus = status;
    }

    else
    {
        bufferGenHdr = bMsg ? ((RM_FLCN_MSG *)pData)->msgGen.hdr :
                                ((RM_FLCN_CMD *)pData)->cmdGen.hdr;
        //
        // If the size of the message in the header is greater than the size of
        // the structure which will hold the message, then log a breakpoint.
        // Copying data more than the structure can hold can lead to buffer overrun
        // on the stack and lead to fatal errors. Logging a breakpoint here will
        // make sure that we can catch this condition in release drivers by looking
        // at the RmJournal.
        //
        // Note: When this happens, we are essentially not purging the message queue
        // so the TAIL pointer will still point to the start of this message.
        // The next time RM gets a new message from Falcon, it will try to purge this
        // message and will keep on looping trying to purge. It will eventually
        // bugcheck, but at least the breakpoint in the logs will point to this bug
        //
        if ((bufferGenHdr.size > pQueueInfo->maxMsgSize) && (bMsg))
        {
            retStatus = LW_ERR_GENERIC;
            LWSWITCH_ASSERT(0);
        }
        //
        // Check the message header to see if the message has a body.  If it does,
        // read it.  It is not considered an error for a message to contain only
        // a header.
        //
        else if (bufferGenHdr.size > RM_FLCN_QUEUE_HDR_SIZE)
        {
            status = _flcnQueueReaderReadBody(device, pFlcn, pQueue, pData, bMsg);
            if (status != LW_OK)
            {
                LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGBODY(pQueue->queueLogId, status);
                retStatus = status;
            }
        }
    }

    //
    // Queue needs to be closed even if there is error in
    // reading header/message above
    //
    status = pQueue->close(device, pFlcn, pQueue, LW_TRUE);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_CLOSING(pQueue->queueLogId, status);

        //
        // Update the retStatus only if there was no error reading
        // header/message earlier.
        //
        if (LW_OK == retStatus)
        {
            retStatus = status;
        }
    }

    return retStatus;
}

/*!
 * @brief Retrieves the next valid header from the queue.
 *
 * This function attempts to read a message header from the message queue. Upon
 * a successful read, the header is be validated and a check is made to see if
 * the header read is the rewind header.  If found, the queue is rewound and
 * another attempt is be made to read a valid header.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   Falcon object pointer
 * @param[in]  pQueue  The queue to read from
 * @param[in]  pData   The buffer to fill-in
 * @param[in]  bMsg    Msg/Cmd
 *
 * @return 'LW_OK'    If a VALID message is read from the message queue.
 * @return 'LW_ERR_GENERIC' Otherwise.
 */
static LW_STATUS
_flcnQueueReaderGetNextHeader
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    void            *pData,
    LwBool           bMsg
)
{
    LW_STATUS        status;
    RM_FLCN_QUEUE_HDR  bufferGenHdr;

    // attempt to read a message header from the message queue
    status = _flcnQueueReaderReadHeader(device, pFlcn, pQueue, pData, bMsg);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGHDR(pQueue->queueLogId, status);
        return LW_ERR_GENERIC;
    }

    bufferGenHdr = bMsg ? ((RM_FLCN_MSG *)pData)->msgGen.hdr :
                            ((RM_FLCN_CMD *)pData)->cmdGen.hdr;

    //
    // If the rewind header is received, rewind the message queue and re-
    // attempt to read a message header.
    //
    if (bufferGenHdr.unitId == RM_FLCN_UNIT_ID_REWIND)
    {
        pQueue->rewind(device, pFlcn, pQueue);
        status = _flcnQueueReaderReadHeader(device, pFlcn, pQueue, pData, bMsg);
        if (status != LW_OK)
        {
            LWSWITCH_PRINT_QUEUE_READER_ERR_READING_MSGHDR(pQueue->queueLogId, status);
            return LW_ERR_GENERIC;
        }
    }

    bufferGenHdr = bMsg ? ((RM_FLCN_MSG *)pData)->msgGen.hdr :
                            ((RM_FLCN_CMD *)pData)->cmdGen.hdr;
    //
    // Validate the header's unit identifier.  This step is performed AFTER the
    // rewind check as an optimization in the event that we did read a rewind
    // message.  In the event of receiving an invalid unit-id, the rewind check
    // would also have failed.
    //
    if (!RM_FLCN_UNIT_ID_IS_VALID(pFlcn->pQueueInfo, bufferGenHdr.unitId))
    {
        LWSWITCH_PRINT_QUEUE_READER_ILWALID_UNITID(pQueue->queueLogId, bufferGenHdr.unitId);
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

/*!
 * @brief Reads the body of a message/command into the buffer.
 *
 * Simply performs a read operation on a previously opened queue in attempt to
 * read a message body.  This function does not make any attempts to interpret
 * the body's data.
 *
 * @param[in]  device   lwswitch device pointer
 * @param[in]  pFlcn    Falcon object pointer
 * @param[in]  pQueue   The queue to read from
 * @param[in]  pData    The buffer to fill-in
 * @param[in]  bMsg     Msg/Cmd
 *
 * @return 'LW_OK'    If a message is read from the message queue.
 * @return 'LW_ERR_GENERIC' Otherwise.
 */
static LW_STATUS
_flcnQueueReaderReadBody
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    void            *pData,
    LwBool           bMsg
)
{
    LwU32      bytesRead;
    LwU32      readSize;
    LW_STATUS  status;
    RM_FLCN_QUEUE_HDR  bufferGenHdr;

    LWSWITCH_ASSERT(!pQueue->isEmpty(device, pFlcn, pQueue));

    bufferGenHdr = bMsg ? ((RM_FLCN_MSG *)pData)->msgGen.hdr :
                            ((RM_FLCN_CMD *)pData)->cmdGen.hdr;

    //
    // The header contains the size to read for the message/command body. Note that
    // size in the header accounts for the size of the header itself.
    //
    readSize = bufferGenHdr.size - RM_FLCN_QUEUE_HDR_SIZE;

    if(bMsg)
    {
        status   = pQueue->pop(device, pFlcn, pQueue, &((RM_FLCN_MSG *)pData)->msgGen.msg,
                                    readSize, &bytesRead);
    }
    else
    {
        status   = pQueue->pop(device, pFlcn, pQueue, &((RM_FLCN_CMD *)pData)->cmdGen.cmd,
                                    readSize, &bytesRead);
    }

    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING(pQueue->queueLogId, status);
        return status;
    }

    //
    // The number of bytes should always be greater than zero in virtue of the
    // fact the queue is known to be non-empty at this point.
    //
    LWSWITCH_ASSERT(bytesRead != 0);

    //
    // Verify that enough data is read to constitute a full message body.
    // Anything less is considered a logic error as it indicates that we are
    // out of sync with the data that's in the queue (ie. we cannot recognize
    // it). This is not expected to occur.
    //
    if (bytesRead != readSize)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING_UNKNOWN_DATA(pQueue->queueLogId, status);
        LWSWITCH_ASSERT(0);
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

/*!
 * @brief Read a message/command header from the given queue.
 *
 * Simply performs a read operation on a previously opened queue in attempt to
 * read a message header.  This function does not make any attempts to
 * interpret or validate the message header
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   Falcon object pointer
 * @param[in]  pQueue  The queue to read from
 * @param[in]  pData   The buffer to fill-in
 * @param[in]  bMsg    Msg/Cmd
 *
 * @return 'LW_OK'    If a message is read from the message queue.
 * @return 'LW_ERR_ILWALID_STATE' If queue is empty.
 * @return 'LW_ERR_GENERIC' Otherwise.
 */
static LW_STATUS
_flcnQueueReaderReadHeader
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    void            *pData,
    LwBool           bMsg
)
{
    LwU32            bytesRead;
    LW_STATUS        status;
    if (pQueue->isEmpty(device, pFlcn, pQueue))
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_STATE;
    }

    LWSWITCH_PRINT_QUEUE_READER_PRINT_HDR_READ_INFO(pQueue->position);

    if(bMsg)
    {
        // read a header's worth of data from the queue
        status = pQueue->pop(
                     device, pFlcn, pQueue, &((RM_FLCN_MSG *)pData)->msgGen.hdr,
                            RM_FLCN_QUEUE_HDR_SIZE, &bytesRead);
    }
    else
    {
        status = pQueue->pop(
                     device, pFlcn, pQueue, &((RM_FLCN_CMD *)pData)->cmdGen.hdr,
                            RM_FLCN_QUEUE_HDR_SIZE, &bytesRead);
    }

    if (status != LW_OK)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING(pQueue->queueLogId, status);
        return status;
    }

    //
    // The number of bytes should always be greater than zero in virtue of the
    // fact the queue is known to be non-empty at this point.
    //
    LWSWITCH_ASSERT(bytesRead != 0);

    //
    // Verify that enough data is read to constitute a full header.  Anything
    // less is considered a logic error as it indicates that we are out of sync
    // with the data that's in the queue (ie. we cannot recognize it). This is
    // not expected to occur.
    //
    if (bytesRead != RM_FLCN_QUEUE_HDR_SIZE)
    {
        LWSWITCH_PRINT_QUEUE_READER_ERR_READING_UNKNOWN_DATA(pQueue->queueLogId, status);
        LWSWITCH_ASSERT(0);
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

void
flcnQueueRdSetupHal
(
    FLCN   *pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    pHal->queueReadData = _flcnQueueReadData_IMPL;
}

