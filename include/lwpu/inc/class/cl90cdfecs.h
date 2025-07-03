/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _cl90cdFecs_h_
#define _cl90cdFecs_h_

/* This file defines parameters for FECS context switch events*/

#define LW_EVENT_BUFFER_FECS_VERSION 2

/*
 * These are the types of context switch events
 * This field gets added to LW_EVENT_BUFFER_FECS_RECORD to specify the sub type of fecs event
 * Do *not* edit these as they are defined to maintain consistency with CheetAh tools
 */
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_SO                  0x00
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_CTXSW_REQ_BY_HOST   0x01
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK              0x02
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK_WFI          0x0a
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK_GFXP         0x0b
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK_CTAP         0x0c
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK_CILP         0x0d
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_SAVE_END            0x03
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_RESTORE_START       0x04
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_CONTEXT_START       0x05
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_SIMPLE_START        0x06
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_SIMPLE_END          0x07
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_ENGINE_RESET        0xfe
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_ILWALID_TIMESTAMP   0xff
#define LW_EVENT_BUFFER_FECS_CTXSWTAG_LAST                LW_EVENT_BUFFER_FECS_EVENTS_CTXSWTAG_ILWALID_TIMESTAMP

/*
 * Bit fields used to enable a particular sub type of event 
 */
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_SO                   LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_SO)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_CTXSW_REQ_BY_HOST    LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_CTXSW_REQ_BY_HOST)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_FE_ACK               LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_FE_ACK)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_SAVE_END             LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_SAVE_END)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_RESTORE_START        LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_RESTORE_START)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_CONTEXT_START        LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_CONTEXT_START)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_SIMPLE_START         LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_SIMPLE_START)
#define LW_EVENT_BUFFER_FECS_BITMASK_CTXSWTAG_SIMPLE_END           LWBIT(LW_EVENT_BUFFER_FECS_CTXSWTAG_SIMPLE_END)

/* context_id is set to this value if fecs info doesn't match a known channel/tsg handle*/
#define LW_EVENT_BUFFER_ILWALID_CONTEXT     0xFFFFFFFF

/* 
 * PID/context_id are set to these values if the data is from another user's
 * client and the current user is not an administrator
 */
#define LW_EVENT_BUFFER_HIDDEN_PID          0x0
#define LW_EVENT_BUFFER_HIDDEN_CONTEXT      0x0

/* 
 * PID/context_id are set to these values if the data is from a kernel client
 * and the data is being read by a user client
 */
#define LW_EVENT_BUFFER_KERNEL_PID          0xFFFFFFFF
#define LW_EVENT_BUFFER_KERNEL_CONTEXT      0xFFFFFFFF

// V1 ------------------------------------------------------------------------
typedef struct
{
    LwU8    tag;                            ///< LW_EVENT_BUFFER_FECS_CTXSWTAG_*
    LwU8    vmid;
    LwU16   seqno;                          ///< used to detect drop 
    LwU32   context_id;                     ///< channel/tsg handle 
    LwU64   pid LW_ALIGN_BYTES(8);          ///< process id
    LwU64   timestamp LW_ALIGN_BYTES(8);
    /* Do *not* edit items above this to maintain consistency with cheetah tools
    Always add to the end of this structure to retain backward compatibility */
} LW_EVENT_BUFFER_FECS_RECORD_V1;

// V2 ------------------------------------------------------------------------
typedef struct
{
    LwU8    tag;                            ///< LW_EVENT_BUFFER_FECS_CTXSWTAG_*
    LwU8    vmid;
    LwU16   seqno;                          ///< used to detect drop 
    LwU32   context_id;                     ///< channel/tsg handle 
    LwU32   pid;                            ///< process id
    LwU16   reserved0;
    LwU8    migGpuInstanceId;
    LwU8    migComputeInstanceId;
    LwU64   timestamp LW_ALIGN_BYTES(8);
    /* Do *not* edit items above this to maintain consistency with cheetah tools
    Always add to the end of this structure to retain backward compatibility */
} LW_EVENT_BUFFER_FECS_RECORD_V2;

typedef LW_EVENT_BUFFER_FECS_RECORD_V1 LW_EVENT_BUFFER_FECS_RECORD_V0;
typedef LW_EVENT_BUFFER_FECS_RECORD_V1 LW_EVENT_BUFFER_FECS_RECORD;
#endif /* _cl90cdFecs_h_ */
