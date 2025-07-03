/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>

#include <utils/lwassert.h>

#include "lwsocket.h"
#include "riscv_config.h"
#include "riscv_gdbstub.h"
#include "riscv_taskdbg.h"
#include "riscv_prv.h"
#include "riscv_porting.h"

#include "tdbg_legacy_messaging.h"
#include "riscv_taskdbg_gdbcmd.h"
#include "riscv_taskdbg_communication.h"

#include "g_riscv_private.h"

#define CHECK_TDBG_RETURN_VALUE_OK(func) \
do { \
    if ((func) != LW_OK) {\
        dprintf("");\
        return LW_ERR_GENERIC;\
    }\
} while (0);

#define CLAMP(x,l,u) ((x)>(u)?(u):(x)<(l)?(l):(x))

struct
{
// Session Unique Identifier here:
    LwU64 xTCB;
// Session Info
    Session session;
    LwBool allocated;
} typedef TaskDebuggerSession;

/* THREAD LOCAL STORAGE */
static TaskDebuggerSession tdbgSessions[LW_RTOS_PTRACE_MAX_TASKS];

/* MAIN THREAD STORAGE */
static LwBool taskDebuggerSessionInitialized = LW_FALSE;
static RiscVInstance *taskDebuggerSessionInstance = NULL;


static void riscvPrintGdbStubSessionInfo(Session *pSession)
{
    const char states[][32] = {"INVALID", "CREATED", "LISTENING", "ATTACHED", "SUSPENDED", "CLOSING", "WAT?"};
    const char exitActions[][16] = {"CONTINUE\n", "HALT\n", "RESTORE(", "WAT?\n"};
    if (pSession == NULL)
    {
        dprintf("pSession is NULL\n");
        return;
    }
    dprintf("pSession->state: %s\n", states[CLAMP(pSession->state, 0, 6)]);
    dprintf("pSession->exitAction: %s", exitActions[CLAMP(pSession->exitAction, 0, 3)]);
    if (pSession->exitAction == RESTORE_PREVIOUS)
    {
        if (pSession->bTaskWasRunning)
            dprintf("continue)\n");
        else
            dprintf("halt)\n");
    }
    if (pSession->rtosTask != 0)
        dprintf("pSession->xTCB: 0x%16"LwU64_fmtx"\n", pSession->rtosTask);
    dprintf("pSession->continuePollMs: %"LwU64_fmtu"\npSession->stepTicks: %"LwU64_fmtu"\n", pSession->continuePollMs, pSession->stepTicks);
    return;
}

LW_STATUS tdbgListSessions(void)
{
    LwU64 i;
    Session *pSession;
    if (taskDebuggerSessionInitialized == LW_FALSE)
    {
        dprintf("Not initialized yet.\n");
        return LW_OK;
    }
    dprintf("RiscVInstance: %s\n",taskDebuggerSessionInstance->name);
    for (i=0; i<LW_RTOS_PTRACE_MAX_TASKS; ++i)
    {
        dprintf("Session[%"LwU64_fmtu"] xTCB: %16"LwU64_fmtx" Allocated: %s\n", i, tdbgSessions[i].xTCB, tdbgSessions[i].allocated?"True":"False");
        pSession = &tdbgSessions[i].session;
        riscvPrintGdbStubSessionInfo(pSession);
    }
    return LW_OK;
}

LW_STATUS tdbgResetDebugger(Session *pSession)
{
    TaskDebuggerPacket cmd;
    LW_STATUS retVal = LW_ERR_ILWALID_STATE;
    cmd.header.pktType = TDBG_CMD_RESET;
    cmd.header.dataSize = 0;
    retVal = riscvSendCommand(pSession, &cmd);
#ifdef RISCV_TDBG_EMEM_MESSAGING
    if (retVal != LW_OK)
    {
        dprintf("Can't communicate to task debugger using queues.\n");
        riscvEmemEmergencyReset();
        retVal = LW_OK;
    }
#endif
    return retVal;
}

static void tdbgDecodeCrashinfo(LwU64 crashinfo, char *out)
{
    const char flag1[8][16] = { "debuggable", "suspended", "bad", "bad", "bad", "bad", "userstep", "divine"};
    const char flag0[8][16] = { "not debuggable", "running", "", "", "", "", "", ""};
    LwU64 mcause2 = crashinfo & 0xFF000000FFFFFFFFULL;
    LwU8 mcause = (LwU8) (crashinfo >> 40);
    LwU8 flags = (LwU8) (crashinfo >> 48);
    LwBool crash = (mcause==255 ? LW_FALSE : LW_TRUE);
    LwBool first = LW_TRUE;
    LwU8 i;

    out[0] = 0;
    for (i=0; i<8; i++)
    {
        if (!first)
        {
            strcat(out, ", ");
        }
        if (flags & (1<<i))
        {
            if (i==1 && crash)
                strcat(out, "crashed");
            else
                strcat(out, flag1[i]);
            if (flag1[i][0] != 0)
                first = LW_FALSE;
            else
                first = LW_TRUE;
        }
        else
        {
            strcat(out, flag0[i]);
            if (flag0[i][0] != 0)
                first = LW_FALSE;
            else
                first = LW_TRUE;
        }
    }
    if (crash)
    {
        char tmp[48];
#if  LWWATCHCFG_IS_PLATFORM(WINDOWS)
        // reduce security for Windows
        sprintf(tmp, "mcause=%u mcause2=%16"LwU64_fmtx"", mcause, mcause2);
#else
        snprintf(tmp, 48, "mcause=%u mcause2=%16"LwU64_fmtx"", mcause, mcause2);
#endif
        strcat(out, tmp);
    }
}

LW_STATUS tdbgListTasks(RiscVInstance *pInstance)
{
    LW_STATUS retVal;
    TaskDebuggerPacket cmd;
    TaskDebuggerPacket msg;
    LwU64 remainingBytes;
    LwU32 bytesCopied;
    LwU8 tasks;
    LwU8 *buffer;
    char stateInfo[128];
    Session dummySession;

    struct
    {
        LwU32 count;
        struct
        {
            LwU8 name[16];
            LwU64 tcb; //casted to LwU64 because its value is not important
            LwU64 pc;
            LwU64 errorinfo;
        } task[LW_RTOS_PTRACE_MAX_TASKS*4];
    } task_list;
    buffer = (LwU8 *) &task_list;

    memset(&dummySession, 0, sizeof(dummySession));
    dummySession.pInstance = pInstance;

    cmd.header.pktType = TDBG_CMD_TASK_LIST;
    cmd.header.dataSize = 0;

    retVal = riscvSendCommand(&dummySession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(&dummySession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType != TDBG_MSG_BULKDATA_HDR)
    {
        if ((msg.header.dataSize == sizeof(msg.data.errcode.status)) && (msg.header.pktType == TDBG_MSG_ACK_FAIL))
        {
            //User error (usually)
            dprintf("pktType == TDBG_MSG_ACK_FAIL, err=%08"LwU64_fmtx"\n", msg.data.errcode.status);
        }
        else
        {
            //Task Debugger protocol error
            dprintf("pktType != TDBG_MSG_BULKDATA_HDR\n");
        }
        return LW_ERR_ILWALID_STATE;
    }
    if (msg.data.bulkdata_hdr.type != TDBG_BULKDATA_TASKLIST)
    {
        dprintf("Bulk Data hdr wrong type\n");
        return LW_ERR_ILWALID_STATE;
    }
    remainingBytes = msg.data.bulkdata_hdr.size;
    if (remainingBytes > sizeof(task_list))
    {
        dprintf("OOPS! Lwwatch compiled with less space for task list (%lu) than RTOS wants to return (%"LwU64_fmtu")!\n",
        (unsigned long) sizeof(task_list), remainingBytes);
    }
    bytesCopied = 0;

    while (remainingBytes != 0)
    {
        cmd.header.pktType = TDBG_CMD_BULKDATA_ACK;
        cmd.header.dataSize = 0;

        retVal = riscvSendCommand(&dummySession, &cmd);
        if (retVal != LW_OK)
            return retVal;
        retVal = riscvRecvMessage(&dummySession, &msg);
        if (retVal != LW_OK)
            return retVal;

        if (msg.header.pktType != TDBG_MSG_BULKDATA)
        {
            dprintf("Unexpected end of BULKDATA transfer\n");
            tdbgResetDebugger(&dummySession);
            return LW_ERR_ILWALID_STATE;
        }

        memcpy(buffer+bytesCopied, msg.data.raw.U8, msg.header.dataSize);
        bytesCopied += msg.header.dataSize;
        remainingBytes -= msg.header.dataSize;
    }
    if (task_list.count > LW_RTOS_PTRACE_MAX_TASKS)
    {
        dprintf("Task list contains %u tasks, maximum of %u supported in lwwatch build. Truncating.\n",
            (LwU32) task_list.count, (LwU32) LW_RTOS_PTRACE_MAX_TASKS);
        task_list.count = LW_RTOS_PTRACE_MAX_TASKS;
    }
    for (tasks = 0; tasks < task_list.count; tasks++)
    {
        stateInfo[0] = 0;
        //0xCCAABB00CCCCCCCLWLL
        // A: debug flags
        // B: mcause
        // C: mcause2
        tdbgDecodeCrashinfo(task_list.task[tasks].errorinfo, stateInfo);

        dprintf("%u: Task: %20s xTCB: "LwU64_FMT" pc:"LwU64_FMT" state: %s\n", (LwU32) tasks,
            task_list.task[tasks].name, task_list.task[tasks].tcb, task_list.task[tasks].pc, stateInfo);
    }
    return LW_OK;
}

LwBool taskDebuggerCheckInstanceSwitch(const RiscVInstance *pInstance)
{
    LwU64 i;
    if (taskDebuggerSessionInitialized == LW_FALSE)
        return LW_TRUE;
    if (taskDebuggerSessionInstance == pInstance)
        return LW_TRUE; // Allow switching to same instance, but do no cleanup.
    for (i=0; i<LW_RTOS_PTRACE_MAX_TASKS; i++)
    {
        if (tdbgSessions[i].session.state != SESSION_ILWALID)
        {
            dprintf("Instance switch disallowed because a session is still active.\nTry 'rv sessions'.");
            return LW_FALSE;
        }
    }
    taskDebuggerSessionInitialized = LW_FALSE;
    taskDebuggerSessionInstance = NULL;
    return LW_TRUE;
}

/*
 * taskDebuggerStub
 * This version does not do non-stop debugging. Each task being debugged
 * is allocated a session.
 * Inputs:
 *  RiscVInstance *pInstance - Which RISC-V core to bind to
 *  LwU64 xTCB - which TCB to attach to.
 *  LwBool bWaitForHalt - should debugger break into task?
 *  LwU64 xWaitForHaltMaxPollingInterval - how often to poll for halt
 */
LW_STATUS taskDebuggerStub(const RiscVInstance *pInstance, LwU64 xTCB, LwBool bWaitForHalt, LwU64 xWaitForHaltMaxPollingInterval)
{
    LwU64 i;
    LW_STATUS ret;
    LwBool bIsHalted = LW_FALSE;
    Session checkStatus;
    LwU64 wait = 1;

    memset(&checkStatus, 0, sizeof(checkStatus));

    if (taskDebuggerSessionInitialized == LW_FALSE)
    {
        memset(tdbgSessions, 0, sizeof(tdbgSessions));
        taskDebuggerSessionInstance = (RiscVInstance *) pInstance;
        taskDebuggerSessionInitialized = LW_TRUE;
    }

    if (bWaitForHalt && (xWaitForHaltMaxPollingInterval < 2))
    {
        dprintf("bWaitForHalt enabled, but xWaitForHaltMaxPollingInterval is 0 or too low.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (pInstance != taskDebuggerSessionInstance)
    {
        dprintf("Task debugger session instance is invalid.\n");
        return LW_ERR_ILWALID_STATE;
    }
    for (i=0; i<LW_RTOS_PTRACE_MAX_TASKS; i++)
    {
        if ((tdbgSessions[i].allocated == LW_FALSE) ||
            (tdbgSessions[i].allocated == LW_TRUE && tdbgSessions[i].xTCB == xTCB))
        {
            break;
        }
    }
    if (i==LW_RTOS_PTRACE_MAX_TASKS)
    {
        dprintf("No free Task Debugger sessions available. Restart lwwatch.\n");
        return LW_ERR_INSUFFICIENT_RESOURCES;
    }
    checkStatus.rtosTask = xTCB;
    checkStatus.pInstance = pInstance;
task_check_halt:
    ret = _tdbgTaskGetSignal(&checkStatus, &bIsHalted, NULL);
    if (ret != LW_OK)
    {
        dprintf("Unable to communicate with RTOS.\n");
        return LW_ERR_ILWALID_OPERATION;
    }
    if (wait == 1)
        dprintf("Task is %s\n", bIsHalted?"halted.":"running.");

    // Session exists, is suspended, we have Wait-for-interrupt enabled, and is task is running
    if (tdbgSessions[i].allocated == LW_TRUE &&
        tdbgSessions[i].session.state == SESSION_SUSPENDED &&
        (bWaitForHalt) &&
        (xWaitForHaltMaxPollingInterval >= 2) &&
        !bIsHalted)
    {
        if (wait == 1)
            dprintf("***** Polling for halt *****\n");
        wait = ((wait * 2) > xWaitForHaltMaxPollingInterval) ? xWaitForHaltMaxPollingInterval : wait * 2;
        wait = (wait > RISCV_RTOS_TIMEOUT_POLL_MS) ? RISCV_RTOS_TIMEOUT_POLL_MS : wait;
        riscvDelay((LwU32)wait);
        goto task_check_halt;
    }

    tdbgSessions[i].allocated = LW_TRUE;
    tdbgSessions[i].xTCB = xTCB;
    tdbgSessions[i].session.rtosTask = xTCB; //this can be cleared on rvGdbStubDeleteSession()
    tdbgSessions[i].session.continuePollMs = xWaitForHaltMaxPollingInterval;

    //
    // If session is invalid, we should collect initial task state.
    // Future improvement: allow scripting for task debugger.
    // This feature was originally implemented for ICD debugger as POC.
    //
    if (tdbgSessions[i].session.state == SESSION_ILWALID)
    {
        tdbgSessions[i].session.exitAction = RESTORE_PREVIOUS;
        tdbgSessions[i].session.bTaskWasRunning = !bIsHalted;
    }

    // TODO: select port
    ret = _taskDebuggerStubInstance(pInstance, &tdbgSessions[i].session, i+1 /* Port offset */);

    return ret;
}
