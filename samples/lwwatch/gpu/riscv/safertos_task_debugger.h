/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RISCV_SAFERTOS_TASK_DEBUGGER_H_
#define _RISCV_SAFERTOS_TASK_DEBUGGER_H_

#include "riscv_prv.h"

////////////////////////////////////////////////////////////////////////////////
/* SYNC POINT - below this line MUST match RTOS debugger.h                    */
////////////////////////////////////////////////////////////////////////////////

//These values are hardcoded for now as we use EMEM command / message queue.
#define CMD_QUEUE_SIZE (0x80)
#define MSG_QUEUE_SIZE (0x80)

// EMEM backdoor number, best to use 0
#define backdoor_no 0

#define HEADERSIZE 8
#define CMD_DATA_MAX_SIZE (CMD_QUEUE_SIZE - HEADERSIZE - 4)
#define MSG_DATA_MAX_SIZE (MSG_QUEUE_SIZE - HEADERSIZE - 4)

struct
{
    struct
    {
        LwU16 magic; //should be TD
        LwU8 pktType;
        LwU8 dummy;
        LwU16 dataSize; // This does NOT include the header.
        LwU16 dummy2;
    } header;
    union // Aligned to 64 bits
    {
        union
        {
            LwU64 U64[(MSG_QUEUE_SIZE - HEADERSIZE - 4)/sizeof(LwU64)];
            LwU32 U32[(MSG_QUEUE_SIZE - HEADERSIZE - 4)/sizeof(LwU32)];
            LwU8  U8 [(MSG_QUEUE_SIZE - HEADERSIZE - 4)/sizeof(LwU8)];
            // 116 of 116 bytes
        } raw;
        // Commands
        struct
        {
            LwU64 target;
            LwU64 src;
            LwU64 size;
        } memR;
        struct
        {
            LwU64 target;
            LwU64 dest;
            LwU64 data64[(MSG_QUEUE_SIZE - HEADERSIZE - 4 - sizeof(LwU64)*2)/sizeof(LwU64)];
            // 8+8+96 of 116 bytes
        } memW;
        struct
        {
            LwU64 target;
            LwU64 size;
        } tcbRW;
        struct
        {
            LwU64 target;
            LwU64 addr;
            LwU64 flags;
        } breakpoint;
        struct
        {
            LwU64 target;
        } ctrl;
        struct
        {
            LwU64 target;
            LwU64 flags;
        } detach;
        struct
        {
            LwU64 target;
            LwU64 ticks;
        } step;
        // Messages
        struct
        {
            LwU64 status;
        } errcode;
        struct
        {
            LwU64 mcause;
            LwU64 mcause2;
        } haltinfo;
        struct
        {
            LwU64 bytes;
        } byteswritten;
        struct
        {
            LwU64 type;
            LwU64 size;
        } bulkdata_hdr;
    } data;
} typedef TaskDebuggerPacket;

// M E S S A G E S
#define TDBG_MSG_ILWALID 0

//MSG_ACK packets
#define TDBG_MSG_ACK_OK 1
#define TDBG_MSG_ACK_FAIL 2

//MSG_BULKDATA and MSG_BULKDATA_HDR
#define TDBG_MSG_BULKDATA_HDR 3
// Bulk data types:
    #define TDBG_BULKDATA_TASKLIST 0x80000001
    #define TDBG_BULKDATA_MEM 0x80000002
    #define TDBG_BULKDATA_TCB 0x80000003

#define TDBG_MSG_BULKDATA 4

//MSG_DATA packets
#define TDBG_MSG_HALTINFO 5
#define TDBG_MSG_BYTESWRITTEN 6

// C O M M A N D S
#define TDBG_CMD_ILWALID 0
#define TDBG_CMD_ATTACH 1
#define TDBG_CMD_DETACH 2
#define TDBG_CMD_TASK_LIST 3
#define TDBG_CMD_HALT 4
#define TDBG_CMD_GO 5
#define TDBG_CMD_USERSTEP 6
#define TDBG_CMD_GETSIGNAL 7
#define TDBG_CMD_SET_BREAKPOINT 8
#define TDBG_CMD_CLEAR_BREAKPOINT 9
#define TDBG_CMD_WRITE_USER 10
#define TDBG_CMD_READ_USER 11
#define TDBG_CMD_READ_TCB 12
#define TDBG_CMD_WRITE_TCB 13
#define TDBG_CMD_WRITE_LOCAL 14
#define TDBG_CMD_CLEAR_STEP 15
#define TDBG_CMD_BULKDATA_ACK 254
#define TDBG_CMD_RESET 255

#define TDBG_CMD_DETACH_RESUME (1<<0)

////////////////////////////////////////////////////////////////////////////////
/* SYNC POINT - above this line MUST match RTOS debugger.h                    */
////////////////////////////////////////////////////////////////////////////////
#endif //_RISCV_SAFERTOS_TASK_DEBUGGER_H_
