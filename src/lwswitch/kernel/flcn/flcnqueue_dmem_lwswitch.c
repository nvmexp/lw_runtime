/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwmisc.h"
#include "common_lwswitch.h"

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "flcn/flcnqueue_lwswitch.h"

#include "rmflcncmdif_lwswitch.h"

/*!
 * @file   flcnqueue_dmem_lwswitch.c
 * @brief  Provides all the fundamental logic for reading/writing queues.
 *
 * Queues are the primary communication mechanism between the RM and various
 * falcon-based engines such as the PMU and Display Falcon.  The RM requests
 * actions by inserting a data packet (command) into a command queue. This
 * generates an interrupt to the falcon which allows it to wake-up and service
 * the request.  Upon completion of the command, the falcon can optionally
 * write an acknowledgment packet (message) into a separate queue designated
 * for RM-bound messages.
 *
 * There are lwrrently two types of queues used:
 *      1) DMEM queues.  The original queue type.  This file contains the routines
 *          specific to DMEM queues.
 *      2) FB Queues   For infomation specific to FB Queues, see the HDR of
 *          flcnqueue_fb.c.
 * File flcnqueue.c has all routines common to both types of queues.
 *
 * Each queue has distinct "head" and "tail" pointers. The "head" pointer marks
 * the position in the queue where the next write operation will take place;
 * the "tail" marks the position of the next read.  When the head and tail
 * pointers are equal, the queue is empty.  When non-equal, data exists in the
 * queue that needs processed.  Queues are always allocated contiguously in the
 * falcon's DMEM.  It may never be assumed that the queue's head pointer will
 * always be greater than the tail pointer.  Such a condition is legal and
 * oclwrs when the head pointer approaches the end of the queue and data must
 * be written at the beginning of the queue to fit.  This is known as a
 * "rewind" condition.  For simplicity, wrapping is not supported.  That is, a
 * single packet cannot wrap around the boundaries of the queue.  The writer of
 * the queue must ensure that this never oclwrs. When the writer determines
 * that a packet won't fit in the end of the queue, it must write a "rewind"
 * command telling the reader to ignore the rest of the queue and look at the
 * beginning of the queue for the next packet.  When the reader finds the
 * rewind packet, it must look to the beginning of the queue to find the packet
 * to read.  The writer is responsible for ensuring that sufficient space will
 * always exist at the end of the queue for the rewind packet.  The writer is
 * also responsible for ensuring that sufficient space exists at the beginning
 * of the queue for the real packet before writing the rewind command.
 * Finally, upon a rewind condition, the writer is also responsible for
 * ensuring that the head pointer never intercepts the tail pointer.  Such a
 * condition indicates that the queue is full, but is completely
 * indistinguishable from the empty condition (in both cases head and tail are
 * equivalent).
 *
 * The general queue insertion algorithm is as follows:
 * @code
 *    if  head >= tail
 *        if  packet_size <= (queue_size - head - rewind_cmd_size)
 *            write packet
 *        else
 *            if  packet_size <= (tail - queue_start - 1)
 *                write rewind command
 *                write packet
 *            else
 *                abort
 *    else
 *        if  packet_size <= (tail - head - 1)
 *            write packet
 *        else
 *            abort
 * @endcode
 *
 * This module provides a basic queue library to support this mechanism. For
 * simplicity, this module makes minimal distinction between command queues and
 * message queues.  It simply provides APIs for opening a queue and performing
 * basic read/write operations.  The only complexity handled here is the
 * rewind operation that is required as the end of a queue is reached during a
 * write operation.  This module handles that case by requiring the write size
 * as a parameter to the "open for write" function.  For the specifics, see
 * @ref flcnQueueOpenWrite.
 *
 * The following diagrams may be used for reference in several of the space
 * callwlations performed by this module. The two most interesting queue states
 * exist when the head pointer is greater than the tail and vice versa.  Head
 * equal to tail is just a sub-case of head greater than tail.
 *
 * <pre>
 *           (head > tail)                     (tail > head)
 *          .-+-+-+-+-+-+-. <-- qSize         .-+-+-+-+-+-+-. <-- qSize
 *          |             |                   |             |
 *          |    free     |                   |    used     |
 *          |             |                   |             |
 *          +-------------+ <-- head          +-------------+ <-- tail
 *          |             |                   |             |
 *          |             |                   |             |
 *          |    used     |                   |    free     |
 *          |             |                   |             |
 *          |             |                   |             |
 *          +-------------+ <-- tail          +-------------+ <-- head
 *          |             |                   |             |
 *          |    free     |                   |    used     |
 *          |             |                   |             |
 *          `-+-+-+-+-+-+-' <-- qOffset       `-+-+-+-+-+-+-' <-- qOffset
 *
 * To be read bottom-to-top (low-address to high-address)
 */

static LW_STATUS _flcnQueueOpenWrite_dmem   (lwswitch_device *device, PFLCN, PFLCNQUEUE pQueue, LwU32 writeSize);
static LW_STATUS _flcnQueuePop_dmem         (lwswitch_device *device, PFLCN, PFLCNQUEUE pQueue, void *pData, LwU32 size, LwU32 *pBytesRead);
static void      _flcnQueueRewind_dmem      (lwswitch_device *device, PFLCN, PFLCNQUEUE pQueue);
static void      _flcnQueuePush_dmem        (lwswitch_device *device, PFLCN, PFLCNQUEUE pQueue, void *pData, LwU32 size);
static LW_STATUS _flcnQueueTailGet_dmem     (lwswitch_device *device, PFLCN pFlcn, PFLCNQUEUE pQueue, LwU32 *pTail);
static LW_STATUS _flcnQueueTailSet_dmem     (lwswitch_device *device, PFLCN pFlcn, PFLCNQUEUE pQueue, LwU32  tail );
static void      _flcnQueueRead_dmem        (lwswitch_device *device, PFLCN pFlcn, PFLCNQUEUE pQueue, LwU32  offset, LwU8 *pDst, LwU32 sizeBytes);
static void      _flcnQueueWrite_dmem       (lwswitch_device *device, PFLCN pFlcn, PFLCNQUEUE pQueue, LwU32  offset, LwU8 *pSrc, LwU32 sizeBytes);
static LW_STATUS _flcnQueueHasRoom_dmem     (lwswitch_device *device, PFLCN pFlcn, PFLCNQUEUE pQueue, LwU32 writeSize, LwBool *pBRewind);

/*!
 * @brief Construct a Falcon Queue object for a DMEM queue.
 *
 * This is a constructor/initialization function for Falcon Queue objects.
 * Callers can choose to either provide a pre-allocated Falcon Queue object or
 * allow this function to perform the allocation.  The former case is more
 * ideal where a collection of queues must be allocated or when static
 * allocation is desired.
 *
 * @param[in]      device    lwswitch device pointer
 * @param[in]      pFlcn     FLCN object pointer
 * @param[in,out]  pQueue    Pointer to the queue to construct.
 *
 * @return 'LW_OK' upon successful construction/initialization.
 * @return 'LW_ERR_ILWALID_POINTER' when pQueue is NULL.
 */
LW_STATUS
flcnQueueConstruct_dmem_lwswitch
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE      *ppQueue,
    LwU32            queueLogId,
    LwU32            queuePhyId,
    LwU32            offset,
    LwU32            queueSize,
    LwU32            cmdHdrSize
)
{
    PFLCNQUEUE  pQueue;
    LW_STATUS   status = LW_OK;

    status = flcnQueueConstruct_common_lwswitch(device, pFlcn,
        ppQueue,        // ppQueue
        queueLogId,     // Logical ID of the queue
        queuePhyId,     // Physical ID of the queue
        offset,         // offset
        queueSize,      // size
        cmdHdrSize);    // cmdHdrSize

    if (status != LW_OK)
    {
        LWSWITCH_ASSERT(status == LW_OK);
        return status;
    }
    if (*ppQueue == NULL)
    {
        LWSWITCH_ASSERT(*ppQueue != NULL);
        return LW_ERR_ILWALID_POINTER;
    }

    pQueue = *ppQueue;

    pQueue->openWrite    = _flcnQueueOpenWrite_dmem;
    pQueue->rewind       = _flcnQueueRewind_dmem;
    pQueue->pop          = _flcnQueuePop_dmem;
    pQueue->push         = _flcnQueuePush_dmem;
    pQueue->tailGet      = _flcnQueueTailGet_dmem;
    pQueue->tailSet      = _flcnQueueTailSet_dmem;
    pQueue->read         = _flcnQueueRead_dmem;
    pQueue->write        = _flcnQueueWrite_dmem;
    pQueue->hasRoom      = _flcnQueueHasRoom_dmem;

    //
    // Command size cannot be larger than queue size / 2.  Otherwise, it is
    // impossible to send two commands back to back if we start from the
    // beginning of the queue.
    //
    pQueue->maxCmdSize   = queueSize / 2;

    return status;
}

/*!
 * @brief Opens a queue for writing.
 *
 * Opens the given command queue for write operations.  Any number of write
 * operations may be performed between a call to 'open' and the subsequent call
 * to 'close'.  However, the full write-size of the entire transaction must be
 * specified when the queue is opened to ensure that the transaction may be
 * written into a contiguous portion of the queue (the falcon ucode does not
 * support wrapping within a single transaction).  This function handles all
 * wrapping/rewinding of the queue as it becomes necessary to find space.
 *
 * @param[in]  device     lwswitch device pointer
 * @param[in]  pFlcn      FLCN object pointer
 * @param[in]  pQueue     The queue to open
 * @param[in]  writeSize  The size (in bytes) of the entire transaction
 *
 * @return 'LW_OK' if the queue is successfully opened.
 * @return 'LW_ERR_INSUFFICIENT_RESOURCES' if there is insufficient queue space
 * @return 'LW_ERR_GENERIC' otherwise.
 * @see    flcnQueuePush
 * @see    flcnQueueRewind
 */
static LW_STATUS
_flcnQueueOpenWrite_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32            writeSize
)
{
    LwBool    bRewind = LW_FALSE;
    LW_STATUS status;
    //
    // Verify that the queue is not already opened.  This is not expected to
    // occur.
    //
    if (pQueue->bOpened)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: unable to open queue (already opened, queueLogId=0x%x).\n",
            __FUNCTION__, pQueue->queueLogId);
        LWSWITCH_ASSERT(0);
        return LW_ERR_GENERIC;
    }

    //
    // Look at the queue's head and tail pointers and determine if enough space
    // exists in the queue for the write.
    //
    status = _flcnQueueHasRoom_dmem(device, pFlcn, pQueue, writeSize, &bRewind);
    if (LW_OK != status)
    {
        if (LW_ERR_INSUFFICIENT_RESOURCES == status)
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: queue is too full to write data (write-size=0x%x).\n",
                __FUNCTION__, writeSize);
        }

        return status;
    }

    //
    // Reaching this point indicates that the queue is successfully opened
    // and sufficient space exists to write the desired data.  Simply set the
    // queue's write position, set the oflag, and mark the queue as "opened".
    //
    (void)pQueue->headGet(device, pFlcn, pQueue, &pQueue->position);
    pQueue->oflag    = FLCNQUEUE_OFLAG_WRITE;
    pQueue->bOpened  = LW_TRUE;

    // finally, rewind the queue if necessary
    if (bRewind)
    {
        pQueue->rewind(device, pFlcn, pQueue);
    }
    return LW_OK;
}

/*!
 * @brief Reads a buffer of data from the given queue.
 *
 * Read a buffer of data from the given queue.  This function does not
 * interpret the data read in any way.  Consequently, it cannot feasibly
 * detect each and every rewind condition that is possible.  For this
 * reason, it is up the to the caller to interpret the returned data and
 * rewind the queue as necessary. This function keeps track of the current
 * read position in queue (set when the queue is opened). To maintain the
 * required DMEM alignment, the queue position is updated with aligned-read
 * size (size rounded-up to the next DMEM alignment).
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcn       FLCN object pointer
 * @param[in]   pQueue      The queue to read from
 * @param[in]   pData       The buffer to write the read data to
 * @param[in]   size        The number of bytes to read
 * @param[out]  pBytesRead  The number of bytes read from the queue
 *
 * @return 'LW_OK' if the read operation is successful. 'LW_ERR_GENERIC' upon
 *         error.  Reading zero bytes from an empty queue is not considered an
 *         error condition.
 *
 * @see flcnQueueRewind
 */
static LW_STATUS
_flcnQueuePop_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    void            *pData,
    LwU32            size,
    LwU32           *pBytesRead
)
{
    LwU32   head;
    LwU32   tail;
    LwU32   used;

    // set the bytes read to zero in case and error oclwrs
    *pBytesRead = 0;

    // ensure the queue is lwrrently opened for read
    if (!QUEUE_OPENED_FOR_READ(pQueue))
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: queue not opened for read (queueLogId=0x%x).\n",
             __FUNCTION__, pQueue->queueLogId);
        LWSWITCH_ASSERT(0);
        return LW_ERR_GENERIC;
    }

    //
    // The callwlations performed in this function are best described using
    // head and tail terminology. The current head pointer values are always
    // used whereas the cached queue position is used for the tail value. This
    // allows read-operations to be transacted without writing the tail pointer
    // for each read.
    //
    (void)pQueue->headGet(device, pFlcn, pQueue, &head);
    tail = pQueue->position;

    // there is no data in the queue when the head and tail are equal
    if (head == tail)
    {
        return LW_OK;
    }

    //
    // Callwlate the used space in the queue (this limits how much can be read).
    // Two cases:
    //     1. When the head is greater than the tail the amount of data in the
    //        queue is defined by the difference between the head and tail
    //        pointers.
    //
    //     2. When the head is less than the tail, a potential rewind condition
    //        exists. In that case, the amount of data that can be read
    //        (without wrapping) is defined as the difference between the
    //        queue's size and the current tail pointer. Note that 'tail' is
    //        absolute so we need to factor-in the starting-offset of the queue.
    //
    if (head > tail)
    {
        used = head - tail;
    }
    else
    {
        used = pQueue->queueOffset + pQueue->queueSize - tail;
    }

    // ensure we only read what is available and no more
    if (size > used)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: suspicious read op - read size > used size. "
            "(queueLogId=0x%x, read size=0x%x, used size=0x%x).\n",
            __FUNCTION__, pQueue->queueLogId, size, used);
        LWSWITCH_ASSERT(0);

        // best thing we can do is cap the read size
        size = used;
    }

    //
    // Copy the data into the output buffer, update the queue's current
    // position, and return the number of bytes that have been read.
    //
    pQueue->read(device, pFlcn, pQueue, tail, pData, size);
    pQueue->position += LW_ALIGN_UP(size, QUEUE_ALIGNMENT);
    *pBytesRead = size;
    return LW_OK;
}

/*!
 * @brief Rewinds a queue back to its starting offset in DMEM.
 *
 * When the queue is opened for "write", this function writes the rewind
 * command to current queue position and updates the queue position to the
 * beginning of the queue. When opened for "read", only the queue position
 * is updated.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  The queue to rewind.
 *
 * @pre  The queue must be opened for prior to calling this function
 * @see  flcnQueueOpenRead
 * @see  flcnQueueOpenWrite
 */
static void
_flcnQueueRewind_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue
)
{
    RM_FLCN_CMD  rewindCmd;
    LwU32        size = 0;

    //
    // Ensure that the queue is opened before continuing. Failure here
    // is never expected.
    //
    if (!pQueue->bOpened)
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: queue not opened (queueLogId=0x%x).\n",
             __FUNCTION__, pQueue->queueLogId);
        LWSWITCH_ASSERT(0);
        return;
    }

    // write the rewind the command when the queue is opened for "write"
    if (QUEUE_OPENED_FOR_WRITE(pQueue))
    {
        // populate the rewind command
        size = pQueue->populateRewindCmd(device, pFlcn, pQueue, &rewindCmd);

        // write out the rewind command
        pQueue->push(device, pFlcn, pQueue, &rewindCmd, size);
    }

    // manually set the queue position back to the beginning of the queue
    pQueue->position = pQueue->queueOffset;
    return;
}

/*!
 * @brief Writes a buffer of data to a queue.
 *
 * Writes a buffer of data to the given command queue.  This function
 * cannot fail since space checks are performed during the call to open to
 * ensure that sufficient space exists in the queue for the data.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  The queue to write to
 * @param[in]  pData   The buffer of data to write
 * @param[in]  size    The number of bytes to write from the buffer
 */
static void
_flcnQueuePush_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    void            *pData,
    LwU32            size
)
{
    // ensure the queue is lwrrently opened for write
    if (!QUEUE_OPENED_FOR_WRITE(pQueue))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: queue not opened for write (queueLogId=0x%x).\n",
            __FUNCTION__, pQueue->queueLogId);
        LWSWITCH_ASSERT(0);
        return;
    }

    // write that data out to the PMU/DPU DMEM
    pQueue->write(device, pFlcn, pQueue, pQueue->position, pData, size);
    pQueue->position += LW_ALIGN_UP(size, QUEUE_ALIGNMENT);
    return;
}

/*!
 * Checks a queue to see if it has room for a writing data of a specific size.
 *
 * @param[in]   device     lwswitch device pointer
 * @param[in]   pFlcn      FLCN object pointer
 * @param[in]   pQueue     The queue to check for space
 * @param[in]   writeSize  The amount of space to check for
 * @param[out]  pBRewind
 *     Set to 'LW_TRUE' when space may be found if the queue is rewound. This
 *     parameter is optional (may be NULL) for callers not interested in
 *     rewind information.
 *
 * @return 'LW_OK' if the queue contains space (has room) for the write.
 *         'LW_ERR_INSUFFICIENT_RESOURCES' if queue is full.
 */
static LW_STATUS
_flcnQueueHasRoom_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32            writeSize,
    LwBool          *pBRewind
)
{
    LwU32  head;
    LwU32  tail;
    LwU32  free = 0;
    LwBool bRewind = LW_FALSE;

    //
    // Align the writeSize up to to the size the buffer will actually take in
    // the queue.
    //
    writeSize = LW_ALIGN_UP(writeSize, QUEUE_ALIGNMENT);

    // retrieve the current queue's head and tail pointers.
    (void)pQueue->headGet(device, pFlcn, pQueue, &head);
    (void)pQueue->tailGet(device, pFlcn, pQueue, &tail);

    //
    // In the case where the head pointer is greater than the tail pointer,
    // callwlate the amount of space in the command queue that may be used
    // before a REWIND command must be written.  Be sure to account for the
    // size of the REWIND command to ensure it can ALWAYS be written.
    //
    if (head >= tail)
    {
        free  = pQueue->queueOffset + pQueue->queueSize - head;
        free -= pQueue->cmdHdrSize;

        //
        // Set the rewind flag to check if space would exist if the queue
        // were rewound.
        //
        if (writeSize > free)
        {
            bRewind = LW_TRUE;
            head    = pQueue->queueOffset;
        }
    }

    //
    // In the event that the head pointer has wrapped around the queue and
    // the tail has no yet caught up, callwlate the amount of space in the
    // command queue that may be used before the head pointer reaches the tail
    // pointer (this can never be allowed to happen).  This condition is also
    // met if a rewind condition is detected above.
    //
    if (head < tail)
    {
        //
        // Subtract off one byte from the free space to guarantee that the tail
        // is never allowed to be equal to the head pointer unless the queue is
        // truly empty.
        //
        free = tail - head - 1;
    }

    // return the rewind flag
    if (pBRewind != NULL)
    {
        *pBRewind = bRewind;
    }

    return (writeSize <= free) ? LW_OK : LW_ERR_INSUFFICIENT_RESOURCES;
}

/*!
 * Retrieve the current tail pointer for given FLCN queue.
 *
 * @param[in]   device  lwswitch device pointer
 * @param[in]   pFlcn   FLCN object pointer
 * @param[in]   pQueue  Pointer to the queue
 * @param[out]  pTail   Pointer to write with the queue's tail value
 *
 * @return 'LW_OK' if the tail value was successfully retrieved.
 * @return 'LW_ERR_GENERIC' otherwise
 */
static LW_STATUS
_flcnQueueTailGet_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32           *pTail
)
{
    LWSWITCH_ASSERT(pFlcn->pQueueInfo != NULL);
    if (RM_FLCN_QUEUEID_IS_COMMAND_QUEUE(pFlcn->pQueueInfo, pQueue->queueLogId))
    {
        return flcnCmdQueueTailGet(device, pFlcn, pQueue, pTail);
    }
    else
    {
        return flcnMsgQueueTailGet(device, pFlcn, pQueue, pTail);
    }
}

/*!
 * Set the tail pointer for the given FLCN queue.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  tail    The desired tail value
 *
 * @return 'LW_OK' if the tail value was successfully set.
 * @return 'LW_ERR_GENERIC' otherwise
 */
static LW_STATUS
_flcnQueueTailSet_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32            tail
)
{
    LWSWITCH_ASSERT(pFlcn->pQueueInfo != NULL);
    if (RM_FLCN_QUEUEID_IS_COMMAND_QUEUE(pFlcn->pQueueInfo, pQueue->queueLogId))
    {
        return flcnCmdQueueTailSet(device, pFlcn, pQueue, tail);
    }
    else
    {
        return flcnMsgQueueTailSet(device, pFlcn, pQueue, tail);
    }
}

/*!
 * Read a buffer of data from FLCN queue.
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcn       FLCN object pointer
 * @param[in]   pQueue      The queue to read from
 * @param[in]   offset      Offset (from the start of DMEM) to start the read
 * @param[out]  pDst        Buffer to store the read-data
 * @param[in]   sizeBytes   The number of bytes to read
 *
 * @return void
 */
static void
_flcnQueueRead_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32            offset,
    LwU8            *pDst,
    LwU32            sizeBytes
)
{
    if (pFlcn->bEmemEnabled)
    {
        flcnableEmemCopyFrom(device, pFlcn->pFlcnable,
                             offset, pDst, sizeBytes, 0);
    }
    else
    {
        if (flcnDmemCopyFrom(device, pFlcn, offset, pDst, sizeBytes, 0)
            != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to copy from flcn DMEM\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
        }
    }
}

/*!
 * Write a buffer of data to a FLCN queue.
 *
 * @param[in]  device     lwswitch device pointer
 * @param[in]  pFlcn      FLCN object pointer
 * @param[in]  pQueue     The queue to write to
 * @param[in]  offset     Offset (from the start of DMEM) to start the write
 * @param[in]  pSrc       Buffer containing the write-data
 * @param[in]  sizeBytes  The number of bytes to write
 */
static void
_flcnQueueWrite_dmem
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    PFLCNQUEUE       pQueue,
    LwU32            offset,
    LwU8            *pSrc,
    LwU32            sizeBytes
)
{
    if (pFlcn->bEmemEnabled)
    {
        flcnableEmemCopyTo(device, pFlcn->pFlcnable,
                           offset, pSrc, sizeBytes, 0);
    }
    else
    {
        if (flcnDmemCopyTo(device, pFlcn, offset, pSrc, sizeBytes, 0)
            != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to copy to flcn DMEM\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
        }
    }
}
