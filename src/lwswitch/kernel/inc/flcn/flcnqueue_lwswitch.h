/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FLCNQUEUE_LWSWITCH_H_
#define _FLCNQUEUE_LWSWITCH_H_

/*!
 * @file      flcnqueue_lwswitch.h
 * @copydoc   flcnqueue_lwswitch.c
 */

#include "lwstatus.h"

struct lwswitch_device;
struct LWSWITCH_TIMEOUT;
struct FLCN;
struct FLCNQUEUE;
union RM_FLCN_MSG;
union RM_FLCN_CMD;

/*!
 * Define the signature of the callback function that FLCN clients must
 * register when sending a FLCN command or registering for FLCN event
 * notification.  Upon completion of the command or upon intercepting an event
 * of a specific type, the callback will be ilwoked passing the completed
 * sequence or event descriptor to the client along with status to indicate if
 * the message buffer was properly populated.
 *
 * @param[in] device    lwswitch_device pointer
 * @param[in] pMsg      Pointer to the received message
 * @param[in] pParams   Pointer to the parameters
 * @param[in] seqDesc   Sequencer descriptor number
 * @param[in] status    Status for command exelwtion result
 */
typedef void (*FlcnQMgrClientCallback)(struct lwswitch_device *, union RM_FLCN_MSG *pMsg, void *pParams, LwU32 seqDesc, LW_STATUS status);

typedef LW_STATUS (*FlcnQueueClose    )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwBool);
typedef LwBool    (*FlcnQueueIsEmpty  )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue);
typedef LW_STATUS (*FlcnQueueOpenRead )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue);
typedef LW_STATUS (*FlcnQueueOpenWrite)(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32);
typedef LW_STATUS (*FlcnQueuePop      )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, void*, LwU32, LwU32 *);
typedef void      (*FlcnQueuePush     )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, void*, LwU32);
typedef void      (*FlcnQueueRewind   )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue);

typedef LW_STATUS (*FlcnQueueHeadGet           )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32 *pHead);
typedef LW_STATUS (*FlcnQueueHeadSet           )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32  head );
typedef LW_STATUS (*FlcnQueueTailGet           )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32 *pTail);
typedef LW_STATUS (*FlcnQueueTailSet           )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32  tail );
typedef void      (*FlcnQueueRead              )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32  offset, LwU8 *pDst, LwU32 sizeBytes);
typedef void      (*FlcnQueueWrite             )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32  offset, LwU8 *pSrc, LwU32 sizeBytes);
typedef LW_STATUS (*FlcnQueueHasRoom           )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32 writeSize, LwBool *pBRewind);
typedef LW_STATUS (*FlcnQueueLock              )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, struct LWSWITCH_TIMEOUT *pTimeout);
typedef LW_STATUS (*FlcnQueueUnlock            )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue);
typedef LwU32     (*FlcnQueuePopulateRewindCmd )(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, union RM_FLCN_CMD *pFlcnCmd);
typedef LW_STATUS (*FlcnQueueElementUseStateClr)(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE *pQueue, LwU32 queuePos);

/*!
 * This structure defines the various flags that may be passed to the queue
 * "open" API.  Read-operations are allowed on queues opened for 'READ';
 * write-operations are allowed when opened for "WRITE".  The specific flag
 * used when a queue is opened defines behavior of the "close" operation.
 */
typedef enum
{
    FLCNQUEUE_OFLAG_READ = 0,
    FLCNQUEUE_OFLAG_WRITE
} FLCNQUEUE_OFLAG;

/*!
 * Contains all fields, attributes, and functions pertaining to Falcon Queues.
 */
typedef struct FLCNQUEUE
{
    FlcnQueueClose              close;
    FlcnQueueIsEmpty            isEmpty;
    FlcnQueueOpenRead           openRead;
    FlcnQueueOpenWrite          openWrite;
    FlcnQueuePop                pop;
    FlcnQueuePush               push;
    FlcnQueueRewind             rewind;

    FlcnQueueHeadGet            headGet;
    FlcnQueueHeadSet            headSet;
    FlcnQueueTailGet            tailGet;
    FlcnQueueTailSet            tailSet;
    FlcnQueueRead               read;
    FlcnQueueWrite              write;
    FlcnQueueHasRoom            hasRoom;
    FlcnQueuePopulateRewindCmd  populateRewindCmd;
    FlcnQueueElementUseStateClr elementUseStateClr;

    /*!
     * When the queue is lwrrently opened for writing, this value stores the
     * current write position.  This allows multiple writes to be streamed into
     * the queue without updating the head pointer for each individual write.
     */
    LwU32                       position;

    /*! The physical DMEM offset where this queue resides/begins. */
    LwU32                       queueOffset;

    /*!
     * The logical queue identifier for the queue which we use to index into
     * the queue structures inside RM.
     */
    LwU32                       queueLogId;

    /*!
     * The physical queue index indicates the index of the queue pertaining to
     * its type. We can use it to index into the head and tail registers of
     * a particular type(CMD or MSG) of queue.
     * For e.g., consider we have 3 command queues and 2 message queues allocated
     * for a particular falcon, their queueLogId and queuePhyId values will be as:
     * <Assuming the command queues are allocated first>
     * CMDQ0 queuePhyId = 0, queueLogId = 0
     * CMDQ1 queuePhyId = 1, queueLogId = 1
     * CMDQ2 queuePhyId = 2, queueLogId = 2
     *
     * MSGQ0 queuePhyId = 0, queueLogId = 3
     * MSGQ1 queuePhyId = 1, queueLogId = 4
     */
    LwU32                       queuePhyId;

    /*! The size of the queue in bytes for DMEM queue, number of entries for FB queue */
    LwU32                       queueSize;

    /*! The size of the command header in bytes. */
    LwU32                       cmdHdrSize;

    /*!
     * Maximum size for each command.
     */
    LwU32                       maxCmdSize;

    /*! The open-flag that was specified when the queue was opened. */
    FLCNQUEUE_OFLAG             oflag;

    /*!
     * 'LW_TRUE' when data is lwrrently being written info the queue (only
     * pertains to command queues).
     */
    LwBool                      bOpened;

    /*!
     * 'LW_TRUE' when locked granting exclusive access the the lock owner.
     */
    LwBool                      bLocked;

} FLCNQUEUE, *PFLCNQUEUE;

/*!
 * @brief Enumeration to represent each discrete sequence state
 *
 * Each sequence stored in the Sequence Table must have a state associated
 * with it to keep track of used vs. available sequences.
 */
typedef enum
{
    /*! Indicates the sequence is not be used and is available */
    FLCN_QMGR_SEQ_STATE_FREE = 0,

    /*!
     * Indicates the sequence has been reserved for a command, but command has
     * not yet been queued in a command queue.
     */
    FLCN_QMGR_SEQ_STATE_PENDING,

    /*!
     * Indicates the sequence has been reserved for a command and has been
     * queued.
     */
    FLCN_QMGR_SEQ_STATE_USED,

    /*!
     * Indicates that an event has oclwrred (shutdown/reset/...) that caused
     * the sequence to be canceled.
     */
    FLCN_QMGR_SEQ_STATE_CANCELED
} FLCN_QMGR_SEQ_STATE;

/*!
 * @brief   Common SEQ_INFO used by all falcons.
 */
typedef struct FLCN_QMGR_SEQ_INFO
{
    /*!
     * The unique identifier used by the FLCN ucode to distinguish sequences.
     * The ID is unique to all sequences lwrrently in-flight but may be reused
     * as sequences are completed by the FLCN.
     */
    LwU8                    seqNum;
    /*!
     * Similar to 'seqNum' but unique for all sequences ever submitted (i.e.
     * never reused).
     */
    LwU32                   seqDesc;
    /*!
     * The state of the sequence (@ref FLCN_QMGR_SEQ_STATE).
     */
    FLCN_QMGR_SEQ_STATE     seqState;
    /*!
     * The client function to be called when the sequence completes.
     */
    FlcnQMgrClientCallback  pCallback;
    /*!
     * Client-specified params that must be provided to the callback function.
     */
    void                   *pCallbackParams;

    /*!
     * CMD Queue associated with this Seq.
     */
    struct FLCNQUEUE       *pCmdQueue;

} FLCN_QMGR_SEQ_INFO, *PFLCN_QMGR_SEQ_INFO;

LW_STATUS flcnQueueConstruct_common_lwswitch(struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE **ppQueue, LwU32 queueId, LwU32 queuePhyId, LwU32 offset, LwU32 queueSize, LwU32 cmdHdrSize);
LW_STATUS flcnQueueConstruct_dmem_lwswitch  (struct lwswitch_device *device, struct FLCN *pFlcn, struct FLCNQUEUE **ppQueue, LwU32 queueId, LwU32 queuePhyId, LwU32 offset, LwU32 queueSize, LwU32 cmdHdrSize);


// Dumping queues for debugging purpose
LW_STATUS flcnRtosDumpCmdQueue_lwswitch(struct lwswitch_device *device, struct FLCN *pFlcn, LwU32 queueLogId, union RM_FLCN_CMD *FlcnCmd);

/*!
 * Alignment to use for all head/tail pointer updates.  Pointers are always
 * rouned up to the nearest multiple of this value.
 */
#define QUEUE_ALIGNMENT  (4)

/*!
 * Checks if the given queue is lwrrently opened for read.
 */
#define QUEUE_OPENED_FOR_READ(pQueue)                                          \
    (((pQueue)->bOpened) && ((pQueue)->oflag == FLCNQUEUE_OFLAG_READ))

/*!
 * Checks if the given queue is lwrrently opened for write.
 */
#define QUEUE_OPENED_FOR_WRITE(pQueue)                                         \
    (((pQueue)->bOpened) && ((pQueue)->oflag == FLCNQUEUE_OFLAG_WRITE))

#endif // _FLCNQUEUE_LWSWITCH_H_

