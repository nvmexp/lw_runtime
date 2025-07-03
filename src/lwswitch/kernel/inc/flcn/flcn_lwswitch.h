/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FLCN_LWSWITCH_H_
#define _FLCN_LWSWITCH_H_

#include "flcn/flcnrtosdebug_lwswitch.h"         // <TODO - HEADER CLEANUP>
#include "flcnifcmn.h"
#include "flcn/flcnqueue_lwswitch.h"

#include "flcn/haldefs_flcn_lwswitch.h"
#include "common_lwswitch.h"

#include "riscvifriscv.h"

/**************** Resource Manager Defines and Structures ******************\
*                                                                           *
* Module: FLCN_LWSWITCH.H                                                   *
*       Defines and structures used for the Falcon Object. The Falcon       *
*       object is the base object for all Falcon-derived engines.           *
\***************************************************************************/

/*!
 * Compares an unit id against the values in the unit_id enumeration and
 * verifies that the id is valid.  It is expected that the id is specified
 * as an unsigned integer.
 */
#define  RM_FLCN_UNIT_ID_IS_VALID(pQeueInfo, id)                                       \
             ((id) < (pQeueInfo)->maxUnitId)


/*!
 * Verifies that the given queue identifier is a valid command queue id.  It
 * is expected that the id is specified as an unsigned integer.
 */
#define  RM_FLCN_QUEUEID_IS_COMMAND_QUEUE(pQeueInfo, id)       \
             ((id)  <= (pQeueInfo)->maxCmdQueueIndex)

/*!
 * Define a sequence descriptor that may be used during initialization that
 * represents an invalid sequence descriptor (one in which will never be
 * assigned when a sequence/command is submitted).
 */
#define FLCN_ILWALID_SEQ_DESC    LW_U32_MAX

/*!
 * Define a event descriptor that may be used during initialization that
 * represents an invalid event descriptor (one in which will never be assigned
 * when a event is registered).
 */
#define FLCN_ILWALID_EVT_DESC    LW_U32_MAX

/*!
 * Defines the alignment/granularity of falcon memory blocks
 */
#define FLCN_BLK_ALIGNMENT (256)

/*!
 * Defines the required address/offset alignment for all DMEM accesses
 */
#define FLCN_DMEM_ACCESS_ALIGNMENT (4)

typedef struct FLCN_EVENT_INFO FLCN_EVENT_INFO, *PFLCN_EVENT_INFO;

/*!
 * @brief Tracks all information for each client that has registered for a
 *        specific type of event-notification.
 */
struct FLCN_EVENT_INFO
{
    /*!
     * A unique identifier given to each event info instance to provide a
     * fast way to identify and track an event registration.
     */
    LwU32                  evtDesc;

    /*!
     * An identifier that describes the type of event the client wants
     * notification of.
     */
    LwU32                  unitId;

    /*!
     * The client's pre-allocated message buffer.  This is the buffer that
     * the message data will be written to when extracted from the Message
     * Queue.  This buffer must be sufficiently sized to hold the largest
     * possible event for type 'unitId'.
     */
    union RM_FLCN_MSG     *pMsg;

    /*! The client function to be called when the event triggers. */
    FlcnQMgrClientCallback pCallback;

    /*!
     * Any client-specified private parameters that must be provided in the
     * callback function.
     */
    void                  *pCallbackParams;

    /*!
     * Client's are tracked as a linked list.  This is a pointer to the next
     * client in the list.  The ordering of this list implies no association
     * between the clients.
     */
    FLCN_EVENT_INFO     *pNext;
};

/*!
 * @brief Enumeration for each discrete command state.
 */
typedef enum FLCN_CMD_STATE
{
    /*!
     * Indicates the the command does not have a state.  Commands/sequences
     * that have never been submitted while possess this state.
     */
    FLCN_CMD_STATE_NONE = 0,

    /*! Indicates that the command is being processed by the FLCN.     */
    FLCN_CMD_STATE_RUNNING,

    /*! Indicates that the command has finished exelwtion on the FLCN. */
    FLCN_CMD_STATE_DONE
} FLCN_CMD_STATE;


typedef struct FALCON_EXTERNAL_CONFIG
{
    LwBool bResetInPmc;                 // If TRUE, Reset Falcon using PMC Enable
    LwU32 riscvRegisterBase;            // Falcon's RISCV base offset.
    LwU32 fbifBase;                     // Falcon's FB Interface base.
    LwU32 blkcgBase;                    // Falcon's BLKCG base.
} FALCON_EXTERNAL_CONFIG, *PFALCON_EXTERNAL_CONFIG;

typedef struct
{
    LwU8                maxUnitId;                  //<! Maximum valid Unit Id
    LwU8                initEventUnitId;            //<! INIT event unit id
    LwU8                cmdQHeadSize;               //<! Command Queue Head Size
    LwU8                cmdQTailSize;               //<! Command Queue Tail size
    LwU8                msgQHeadSize;               //<! Message Queue Head Size
    LwU8                msgQTailSize;               //<! Message Queue Tail Size
    LwU32               maxCmdQueueIndex;           //<! Maximum command queue Index
    LwU32               maxMsgSize;                 //<! Maximum valid MSG size
    LwU32               cmdQHeadBaseAddress;        //<! Base Register Address of Command Queue Head.
    LwU32               cmdQHeadStride;             //<! Stride used to access indexed Command Queue Head registers.
    LwU32               cmdQTailBaseAddress;        //<! Base Register Address of Command Queue Tail.
    LwU32               cmdQTailStride;             //<! Stride used to access indexed Command Queue Tail registers.
    LwU32               msgQHeadBaseAddress;        //<! Base Register Address of Message Queue Head.
    LwU32               msgQHeadStride;             //<! Stride used to access indexed Message Queue Head registers.
    LwU32               msgQTailBaseAddress;        //<! Base Register Address of Message Queue Tail.
    LwU32               msgQTailStride;             //<! Stride used to access indexed Message Queue Head registers.
    FLCNQUEUE          *pQueues;                    //<! Queues allocated

    /*!
     * A linked-list of event information structures tracking all clients that
     * have registered for event notification.
     */
    PFLCN_EVENT_INFO pEventInfo;

    /*!
     * Each time a client registers for an event notification, an internal
     * data structure is created and attached to the event-info list.  Each
     * structure will have a unique identifier/descriptor assigned that will
     * be used to track and manage the registration.  This variable keeps track
     * of the next descriptor that will be assigned at any given time.
     */
    LwU32               nextEvtDesc;

    /*!
     * Similar to 'nextEvtDesc' keeps track of the command descriptor that
     * will be assigned to the next queued command.
     */
    LwU32               nextSeqDesc;

    /*!
     * Keeps track the latest used sequence number. We always search the free
     * sequence starting from the next to the latest used sequence since it is
     * the most possible free sequence if we consume the sequence in serial.
     */
    LwU32               latestUsedSeqNum;

} FALCON_QUEUE_INFO,
*PFALCON_QUEUE_INFO;


/*!
 * Data specific Falcon debugging features.
 */
typedef struct
{
    LwU32           dumpEngineTag;        // LWDUMP_COMPONENT_ENG_xyz.
    LwU32           pbFalconId;           // Protobuf falcon ID.  RTOS_FLCN_xyz.
    LwU16           debugInfoDmemOffset;  // DMEM address of the Falcon's
                                          // DEBUG_INFO structure.
    LwBool          bCrashed;             // Falcon has crashed at least once
                                          // since RM was initialized.
    LwBool          bCallbackTriggered;   // Flag indicating that callback
                                          // was actually called.
} FLCN_DEBUG, *PFLCN_DEBUG;

struct FLCNABLE;

typedef struct ENGINE_DESCRIPTOR_TYPE
{
    LwU32   base;
    LwBool  initialized;
} ENGINE_DESCRIPTOR_TYPE, *PENGINE_DESCRIPTOR_TYPE;

typedef enum ENGINE_TAG
{
    ENG_TAG_ILWALID,
    ENG_TAG_SOE,
    ENG_TAG_END_ILWALID
} ENGINE_TAG, *PENGINE_TAG;

typedef struct FLCN
{
    // pointer to our function table - should always be the first thing in any object
    flcn_hal *pHal;

    // we don't have a parent class, so we go straight to our members
    const char             *name;

    ENGINE_DESCRIPTOR_TYPE engDeslwc;
    ENGINE_DESCRIPTOR_TYPE engDescBc;

    FALCON_EXTERNAL_CONFIG extConfig;

    //
    // State variables
    //
    LwBool                bConstructed;

    /*! The FLCN is ready to accept work from the RM. */
    LwBool                 bOSReady;

    /*! This Falcon will have queue support */
    LwBool                bQueuesEnabled;
    LwU8                  numQueues;         //<! Number of queues constructed
    LwU32                 numSequences;      //<! Number of sequences constructed

    FLCN_DEBUG            debug;             //<! Data specific to debugging
    LwU8                  coreRev;           //<! Core revision.  0x51 is 5.1.
    LwU8                  selwrityModel;     //<! Follows _FALCON_HWCFG1_SELWRITY_MODEL_xyz
    // Replacement for a PDB Property: PDB_PROP_FLCN_SUPPORTS_DMEM_APERTURES
    LwBool                supportsDmemApertures;

    // We need to save a pointer to the FLCNABLE interface
    struct FLCNABLE      *pFlcnable;

    ENGINE_TAG            engineTag;

    PFALCON_QUEUE_INFO    pQueueInfo;

    /*!
     * Determines whether to use EMEM in place of DMEM for RM queues and
     * the RM managed heap. EMEM is a memory region outside of the core engine
     * of some falcons which allows for RM access even when the falcon is
     * locked down in HS mode. This is required so that engines like SEC2
     * can receive new commands from RM without blocking.
     */
    LwBool                bEmemEnabled;

    /*! HW arch that is enabled and running on corresponding uproc engine. */
    LwU32                 engArch;
} FLCN, *PFLCN;

// hal functions

// OBJECT Interfaces
LW_STATUS flcnQueueReadData(struct lwswitch_device *, PFLCN, LwU32 queueId, void *pData, LwBool bMsg);
LW_STATUS flcnQueueCmdWrite(struct lwswitch_device *, PFLCN, LwU32 queueId, union RM_FLCN_CMD *pCmd, struct LWSWITCH_TIMEOUT *pTimeout);
LW_STATUS flcnQueueCmdCancel(struct lwswitch_device *, PFLCN, LwU32 seqDesc);
LW_STATUS flcnQueueCmdPostNonBlocking(struct lwswitch_device *, PFLCN, union RM_FLCN_CMD *pCmd, union RM_FLCN_MSG *pMsg, void *pPayload, LwU32 queueIdLogical, FlcnQMgrClientCallback pCallback, void *pCallbackParams, LwU32 *pSeqDesc, struct LWSWITCH_TIMEOUT *pTimeout);
LW_STATUS flcnQueueCmdPostBlocking(struct lwswitch_device *, PFLCN, union RM_FLCN_CMD *pCmd, union RM_FLCN_MSG *pMsg, void *pPayload, LwU32 queueIdLogical, LwU32 *pSeqDesc, struct LWSWITCH_TIMEOUT *pTimeout);
LW_STATUS flcnQueueCmdWait(struct lwswitch_device *, PFLCN, LwU32, struct LWSWITCH_TIMEOUT *pTimeout);
LwU8 flcnCoreRevisionGet(struct lwswitch_device *, PFLCN);
void flcnMarkNotReady(struct lwswitch_device *, PFLCN);
LW_STATUS flcnCmdQueueHeadGet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 *pHead);
LW_STATUS flcnMsgQueueHeadGet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 *pHead);
LW_STATUS flcnCmdQueueTailGet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 *pTail);
LW_STATUS flcnMsgQueueTailGet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 *pTail);
LW_STATUS flcnCmdQueueHeadSet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 head);
LW_STATUS flcnMsgQueueHeadSet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 head);
LW_STATUS flcnCmdQueueTailSet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 tail);
LW_STATUS flcnMsgQueueTailSet(struct lwswitch_device *, PFLCN, FLCNQUEUE *pQueue, LwU32 tail);
PFLCN_QMGR_SEQ_INFO flcnQueueSeqInfoFind(struct lwswitch_device *, PFLCN, LwU32 seqDesc);
PFLCN_QMGR_SEQ_INFO flcnQueueSeqInfoAcq(struct lwswitch_device *, PFLCN);
void flcnQueueSeqInfoRel(struct lwswitch_device *, PFLCN, PFLCN_QMGR_SEQ_INFO pSeqInfo);
void flcnQueueSeqInfoStateInit(struct lwswitch_device *, PFLCN);
void flcnQueueSeqInfoCancelAll(struct lwswitch_device *, PFLCN);
LW_STATUS flcnQueueSeqInfoFree(struct lwswitch_device *, PFLCN, PFLCN_QMGR_SEQ_INFO);
LW_STATUS flcnQueueEventRegister(struct lwswitch_device *, PFLCN, LwU32 unitId, LwU8 *pMsg, FlcnQMgrClientCallback pCallback, void *pParams, LwU32 *pEvtDesc);
LW_STATUS flcnQueueEventUnregister(struct lwswitch_device *, PFLCN, LwU32 evtDesc);
LW_STATUS flcnQueueEventHandle(struct lwswitch_device *, PFLCN, union RM_FLCN_MSG *pMsg, LW_STATUS evtStatus);
LW_STATUS flcnQueueResponseHandle(struct lwswitch_device *, PFLCN, union RM_FLCN_MSG *pMsg);
LwU32 flcnQueueCmdStatus(struct lwswitch_device *, PFLCN, LwU32 seqDesc);
LW_STATUS flcnDmemCopyFrom(struct lwswitch_device *, PFLCN, LwU32 src, LwU8 *pDst, LwU32 sizeBytes, LwU8 port);
LW_STATUS flcnDmemCopyTo(struct lwswitch_device *, PFLCN, LwU32 dst, LwU8 *pSrc, LwU32 sizeBytes, LwU8 port);
void flcnPostDiscoveryInit(struct lwswitch_device *, PFLCN);
void flcnDbgInfoDmemOffsetSet(struct lwswitch_device *, PFLCN, LwU16 debugInfoDmemOffset);

// HAL Interfaces
LW_STATUS   flcnConstruct_HAL                           (struct lwswitch_device *, PFLCN);
void        flcnDestruct_HAL                            (struct lwswitch_device *, PFLCN);
LwU32       flcnRegRead_HAL                             (struct lwswitch_device *, PFLCN, LwU32 offset);
void        flcnRegWrite_HAL                            (struct lwswitch_device *, PFLCN, LwU32 offset, LwU32 data);
const char *flcnGetName_HAL                             (struct lwswitch_device *, PFLCN);
LwU8        flcnReadCoreRev_HAL                         (struct lwswitch_device *, PFLCN);
void        flcnGetCoreInfo_HAL                         (struct lwswitch_device *, PFLCN);
LW_STATUS   flcnDmemTransfer_HAL                        (struct lwswitch_device *, PFLCN, LwU32 src, LwU8 *pDst, LwU32 sizeBytes, LwU8 port, LwBool bCopyFrom);
void        flcnIntrRetrigger_HAL                       (struct lwswitch_device *, PFLCN);
LwBool      flcnAreEngDescsInitialized_HAL              (struct lwswitch_device *, PFLCN);
LW_STATUS   flcnWaitForResetToFinish_HAL                (struct lwswitch_device *, PFLCN);
void        flcnDbgInfoCapturePcTrace_HAL               (struct lwswitch_device *, PFLCN);
void        flcnDbgInfoCaptureRiscvPcTrace_HAL          (struct lwswitch_device *, PFLCN);
LwU32       flcnDmemSize_HAL                            (struct lwswitch_device *, struct FLCN *);
LwU32       flcnSetImemAddr_HAL                         (struct lwswitch_device *, struct FLCN *, LwU32 dst);
void        flcnImemCopyTo_HAL                          (struct lwswitch_device *, struct FLCN *, LwU32 dst, LwU8 *pSrc, LwU32 sizeBytes, LwBool bSelwre, LwU32 tag, LwU8 port);
LwU32       flcnSetDmemAddr_HAL                         (struct lwswitch_device *, struct FLCN *, LwU32 dst);
LwU32       flcnRiscvRegRead_HAL                        (struct lwswitch_device *, PFLCN, LwU32 offset);
void        flcnRiscvRegWrite_HAL                       (struct lwswitch_device *, PFLCN, LwU32 offset, LwU32 data);

// Falcon core revision / subversion definitions.
#define LW_FLCN_CORE_REV_3_0    0x30  // 3.0 - Core revision 3 subversion 0.
#define LW_FLCN_CORE_REV_4_0    0x40  // 4.0 - Core revision 4 subversion 0.
#define LW_FLCN_CORE_REV_4_1    0x41  // 4.1 - Core revision 4 subversion 1.
#define LW_FLCN_CORE_REV_5_0    0x50  // 5.0 - Core revision 5 subversion 0.
#define LW_FLCN_CORE_REV_5_1    0x51  // 5.1 - Core revision 5 subversion 1.
#define LW_FLCN_CORE_REV_6_0    0x60  // 6.0 - Core revision 6 subversion 0.

//
// Colwert Falcon core rev/subver to the IP version format that can be recognized
// by the chip-config dynamic HAL infra.
//
#define LW_FLCN_CORE_REV_TO_IP_VER(coreVer)                       \
    (DRF_NUM(_PFALCON, _IP_VER, _MAJOR, ((coreVer >> 4) & 0xf)) | \
     DRF_NUM(_PFALCON, _IP_VER, _MINOR, (coreVer & 0xf)))

#define LW_PFALCON_IP_VER_MINOR                                                     23:16
#define LW_PFALCON_IP_VER_MAJOR                                                     31:24

// Some mailbox defines (should be shared with MSDEC OS)
#define LW_FALCON_MAILBOX0_MSDECOS_STATUS                   11:0
#define LW_FALCON_MAILBOX0_MSDECOS_ILWALID_METHOD_MTHDCNT   19:12
#define LW_FALCON_MAILBOX0_MSDECOS_ILWALID_METHOD_MTHDID    31:20
#define LW_FALCON_MAILBOX1_MSDECOS_ILWALID_METHOD_MTHDDATA  31:0

PFLCN flcnAllocNew(void);
LwlStatus flcnInit(lwswitch_device *device, PFLCN pFlcn, LwU32 pci_device_id);
void flcnDestroy(lwswitch_device *device, FLCN *pFlcn);

/*!
 * The HW arch (e.g. FALCON or FALCON + RISCV) that can be actively enabled and
 * running on an uproc engine.
 */
#define LW_UPROC_ENGINE_ARCH_DEFAULT        (0x0)
#define LW_UPROC_ENGINE_ARCH_FALCON         (0x1)
#define LW_UPROC_ENGINE_ARCH_FALCON_RISCV   (0x2)

/*!
 * Hepler macro to check what HW arch is enabled and running on an uproc engine.
 */
#define UPROC_ENG_ARCH_FALCON(pFlcn)        (pFlcn->engArch == LW_UPROC_ENGINE_ARCH_FALCON)
#define UPROC_ENG_ARCH_FALCON_RISCV(pFlcn)  (pFlcn->engArch == LW_UPROC_ENGINE_ARCH_FALCON_RISCV)

// Falcon Register index
#define LW_FALCON_REG_R0                       (0)
#define LW_FALCON_REG_R1                       (1)
#define LW_FALCON_REG_R2                       (2)
#define LW_FALCON_REG_R3                       (3)
#define LW_FALCON_REG_R4                       (4)
#define LW_FALCON_REG_R5                       (5)
#define LW_FALCON_REG_R6                       (6)
#define LW_FALCON_REG_R7                       (7)
#define LW_FALCON_REG_R8                       (8)
#define LW_FALCON_REG_R9                       (9)
#define LW_FALCON_REG_R10                      (10)
#define LW_FALCON_REG_R11                      (11)
#define LW_FALCON_REG_R12                      (12)
#define LW_FALCON_REG_R13                      (13)
#define LW_FALCON_REG_R14                      (14)
#define LW_FALCON_REG_R15                      (15)
#define LW_FALCON_REG_IV0                      (16)
#define LW_FALCON_REG_IV1                      (17)
#define LW_FALCON_REG_UNDEFINED                (18)
#define LW_FALCON_REG_EV                       (19)
#define LW_FALCON_REG_SP                       (20)
#define LW_FALCON_REG_PC                       (21)
#define LW_FALCON_REG_IMB                      (22)
#define LW_FALCON_REG_DMB                      (23)
#define LW_FALCON_REG_CSW                      (24)
#define LW_FALCON_REG_CCR                      (25)
#define LW_FALCON_REG_SEC                      (26)
#define LW_FALCON_REG_CTX                      (27)
#define LW_FALCON_REG_EXCI                     (28)
#define LW_FALCON_REG_RSVD0                    (29)
#define LW_FALCON_REG_RSVD1                    (30)
#define LW_FALCON_REG_RSVD2                    (31)

#define LW_FALCON_REG_SIZE                     (32)

#define FALC_REG(x)                            LW_FALCON_REG_##x


#endif // _FLCN_LWSWITCH_H_

/*!
 * Defines the Falcon IMEM block-size (as a power-of-2).
 */
#define FALCON_IMEM_BLKSIZE2 (8)

/*!
 * Defines the Falcon DMEM block-size (as a power-of-2).
 */
#define FALCON_DMEM_BLKSIZE2 (8)

