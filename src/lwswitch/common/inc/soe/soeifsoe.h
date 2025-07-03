/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOEIFSOE_H_
#define _SOEIFSOE_H_

#include "riscvifriscv.h"

/*!
 * @file   soeifsoe.h
 * @brief  SOE Command/Message Interfaces - SOE Management
 */
#ifndef AES128_BLOCK_SIZE_BYTES
#define AES128_BLOCK_SIZE_BYTES 16
#endif
/*!
 * Defines the identifiers various high-level types of sequencer commands and
 * messages.
 *
 * _SOE_INIT @ref RM_SOE_INIT_MSG_SOE_INIT
 */
enum
{
    RM_SOE_INIT_MSG_ID_SOE_INIT = 0,
};


/*!
 * Defines the logical queue IDs that must be used when submitting commands
 * to or reading messages from SOE. The identifiers must begin with zero and
 * should increment sequentially. _CMDQ_LOG_ID__LAST must always be set to the
 * last command queue identifier. _NUM must always be set to the last identifier
 * plus one.
 */
#define SOE_RM_CMDQ_LOG_ID       0
#define SOE_RM_CMDQ_LOG_ID__LAST 0
#define SOE_RM_MSGQ_LOG_ID       1
#define SOE_QUEUE_NUM            2

/*!
 * Message sent from SOE to the RM upon initialization as an event to
 * indicate that SOE is initialized and "ready" to process commands from
 * the RM.
 */
typedef struct
{
    LwU8  msgType;
    LwU8  numQueues;

    LwU16 osDebugEntryPoint;   //<! DMEM address of SOE's DEBUG_INFO

    /*!
     * SOE command and message queue locations and sizes are determined at SOE
     * build-time. Each queue is represented by a single element in this array.
     *
     * When EMEM support is enabled, the offsets for queues and the
     * rmManagedArea will be DMEM addresses located in EMEM, which is
     * mapped right on top of the DMEM VA space.
     */
    struct
    {
        LwU32 queueOffset;      //<! DMEM address of the start of the queue
        LwU16 queueSize;        //<! Size of the queue (in bytes)
        LwU8  queuePhyId;       //<! Physical/register-index of the queue
        LwU8  queueLogId;       //<! Logical ID of the queue
    } qInfo[SOE_QUEUE_NUM];

    LwU32  rmManagedAreaOffset;                         //<! DMEM address of the RM-managed area
    LwU16  rmManagedAreaSize;                           //<! Size (in bytes) of the RM-managed area
    LwU8   devidEnc[AES128_BLOCK_SIZE_BYTES];           //<! Encrypted DEVID for devid name lookup
    LwU8   devidDerivedKey[AES128_BLOCK_SIZE_BYTES];    //<! Derived key used by RM for further decryption of devid name

    FLCN_STATUS status;

} RM_SOE_INIT_MSG_SOE_INIT;

typedef union
{
    LwU8                      msgType;
    RM_SOE_INIT_MSG_SOE_INIT  soeInit;
} RM_FLCN_MSG_SOE_INIT;

/*!
 * Boot arguments for RISC-V SEC2. It contains RISC-V and legacy "command line".
 */
typedef struct
{
    LW_RISCV_BOOT_PARAMS riscv;
} RM_SOE_BOOT_PARAMS, *PRM_SOE_BOOT_PARAMS;

#endif  // _SOEIFSOE_H_
