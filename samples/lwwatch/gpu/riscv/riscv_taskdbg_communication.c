/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <utils/lwassert.h>

#include "riscv_gdbstub.h"
#include "riscv_prv.h"

#include "tdbg_legacy_messaging.h"

#include "riscv_taskdbg_communication.h"
#include "riscv_gdbstub_common.h"

// Caller allocate.
LW_STATUS _tdbgTaskMemRead(Session *pSession, LwU8 *pBuffer, LwU64 addr, unsigned *pSize)
{
    unsigned size;
    LW_STATUS retVal;
    TaskDebuggerPacket cmd;
    TaskDebuggerPacket msg;
    LwU64 remainingBytes;
    LwU32 bytesCopied;

    size = *pSize;
    cmd.header.pktType = TDBG_CMD_READ_USER;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.memR);
    cmd.data.memR.target = pSession->rtosTask;
    cmd.data.memR.src = addr;
    cmd.data.memR.size = size;

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType != TDBG_MSG_BULKDATA_HDR)
    {
        if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
        {
            //User error (usually), or memory address bad (as in no access at all)
            dprintf("_tdbgTaskMemRead: Memory at 0x%16"LwU64_fmtx" inaccessible.\n", addr);
        }
        else
        {
            //Task Debugger protocol error
            dprintf("_tdbgTaskMemRead: protocol violation\n");
        }
        return LW_ERR_ILWALID_STATE;
    }
    if (msg.data.bulkdata_hdr.type != TDBG_BULKDATA_MEM)
    {
        dprintf("_tdbgTaskMemRead: Bulk Data hdr wrong type\n");
        return LW_ERR_ILWALID_STATE;
    }
    remainingBytes = msg.data.bulkdata_hdr.size;
    if (remainingBytes < size) // Task cannot read all of requested memory.
    {
        dprintf("_tdbgTaskMemRead: Warning, read incomplete!\n");
        // Let calling function know we had problem.
        *pSize = (unsigned) remainingBytes;
    }
    else if (remainingBytes > size)
    {
        dprintf("_tdbgTaskMemRead: Error, task wants to return larger value than allocated buffer\n");
        // Any time we have to abort bulkdata transfer we need to reset debugger.
        tdbgResetDebugger(pSession);
        return LW_ERR_ILWALID_STATE;
    }

    memset(pBuffer, 0xee, size); //pad with dummy values

    bytesCopied = 0;
    while (remainingBytes != 0)
    {
        cmd.header.pktType = TDBG_CMD_BULKDATA_ACK;
        cmd.header.dataSize = 0;

        retVal = riscvSendCommand(pSession, &cmd);
        if (retVal != LW_OK)
            return retVal;
        retVal = riscvRecvMessage(pSession, &msg);
        if (retVal != LW_OK)
            return retVal;

        if (msg.header.pktType != TDBG_MSG_BULKDATA)
        {
            dprintf("Unexpected end of BULKDATA transfer\n");
            tdbgResetDebugger(pSession);
            return LW_ERR_ILWALID_STATE;
        }
        memcpy(pBuffer+bytesCopied, msg.data.raw.U8, msg.header.dataSize);
        bytesCopied += msg.header.dataSize;
        remainingBytes -= msg.header.dataSize;
    }
    return LW_OK;
}

// Caller allocate.
LW_STATUS _tdbgTaskMemWritePacket(Session *pSession, LwU8 *pBuffer, LwU64 addr, unsigned *pSize)
{
    unsigned size;
    LW_STATUS retVal;
    TaskDebuggerPacket cmd;
    TaskDebuggerPacket msg;

    size = *pSize;
    // 128 byte command, 8 byte header, 16 byte target/dest, 4 byte command head/tail separation == 100 -> 96 bytes.
    if (size > sizeof(cmd.data.memW.data64) - 4)
    {
        dprintf("_tdbgTaskMemWrite: Error, size (%u) > sizeof(cmd.data.memW.data64[] - 4) (%lu)\n",
                size, (unsigned long)(sizeof(cmd.data.memW.data64) - 4));
        return LW_ERR_ILWALID_ARGUMENT;
    }

    cmd.header.pktType = TDBG_CMD_WRITE_USER;
    cmd.header.dataSize = (LwU16)(size + sizeof(cmd.data.memW.target) + sizeof(cmd.data.memW.dest));
    cmd.data.memW.target = pSession->rtosTask;
    cmd.data.memW.dest = addr;
    memcpy(cmd.data.memW.data64, pBuffer, size);

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType != TDBG_MSG_BYTESWRITTEN)
    {
        if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
        {
            //User error (usually), or memory address bad (as in no access at all)
            dprintf("_tdbgTaskMemWrite: Memory at 0x%16"LwU64_fmtx" inaccessible.\n", addr);
        }
        else
        {
            //Task Debugger protocol error
            dprintf("_tdbgTaskMemWrite: protocol violation\n");
        }
        return LW_ERR_ILWALID_STATE;
    }

    if (msg.data.byteswritten.bytes < size) // Task cannot write all of requested memory
    {
        dprintf("_tdbgTaskMemWrite: Warning, write incomplete!\n");
        // Let calling function know we had problem.
        *pSize = (unsigned) msg.data.byteswritten.bytes;
    }
    else if (msg.data.byteswritten.bytes > size)
    {
        dprintf("Error, bytes written larger than amount sent...?\n");
        tdbgResetDebugger(pSession);
        return LW_ERR_ILWALID_STATE;
    }
    return LW_OK;
}

LW_STATUS _tdbgTaskSetBreakpoint(Session *pSession, const char *pRequest)
{
    TaskDebuggerPacket cmd;
    LwU64 addr;
    TRIGGER_EVENT flags = rvGdbStubMapBreakpointToTrigger(pRequest[0]);

    if (flags == TRIGGER_UNUSED)
        return LW_ERR_ILWALID_ARGUMENT;

    pRequest = strstr(pRequest, ",");
    if (*pRequest)
        pRequest++;
    if (!sscanf(pRequest, "%"LwU64_fmtx"", &addr))
        return LW_ERR_ILWALID_ARGUMENT;

    cmd.header.pktType = TDBG_CMD_SET_BREAKPOINT;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.breakpoint);
    cmd.data.breakpoint.target = pSession->rtosTask;
    cmd.data.breakpoint.addr = addr;
    //
    // TRIGGER_EVENT only sets lower 3 bits. The task debugger should set USER
    // privilege for us (as it's only allowed privilege for TDBG breakpoints).
    //
    cmd.data.breakpoint.flags = flags;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskClearBreakpoint(Session *pSession, const char *pRequest)
{
    TaskDebuggerPacket cmd;
    LwU64 addr;
    TRIGGER_EVENT flags = rvGdbStubMapBreakpointToTrigger(pRequest[0]);

    if (flags == TRIGGER_UNUSED)
        return -1;

    pRequest = strstr(pRequest, ",");
    if (*pRequest)
        pRequest++;
    if (!sscanf(pRequest, "%"LwU64_fmtx"", &addr))
        return LW_ERR_ILWALID_ARGUMENT;

    cmd.header.pktType = TDBG_CMD_CLEAR_BREAKPOINT;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.breakpoint);
    cmd.data.breakpoint.target = pSession->rtosTask;
    cmd.data.breakpoint.addr = addr;
    //
    // ICD debugger uses smart clearing for breakpoints, ie. if clear bits != 0,
    // keep breakpoint enabled. I don't believe GDB wants this behaviour, so
    // clear breakpoint regardless of flags.
    //
    cmd.data.breakpoint.flags = 0;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskHalt(Session *pSession)
{
    TaskDebuggerPacket cmd;
    cmd.header.pktType = TDBG_CMD_HALT;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.ctrl);
    cmd.data.ctrl.target = pSession->rtosTask;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskGetSignal(Session *pSession, LwBool *bIsHalted, LwU64 *cause)
{
    LW_STATUS retVal = LW_ERR_GENERIC;
    TaskDebuggerPacket cmd, msg;

    cmd.header.pktType = TDBG_CMD_GETSIGNAL;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.ctrl);
    cmd.data.ctrl.target = pSession->rtosTask;

    *bIsHalted = LW_FALSE;

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType == TDBG_MSG_ACK_OK) // Did not halt.
    {
        *bIsHalted = LW_FALSE;
        // Do not touch cause.
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_HALTINFO)
    {
        *bIsHalted = LW_TRUE;
        if (cause != NULL)
        {
            if (msg.data.haltinfo.mcause != -1) // Do not touch cause if returning "no signal"
            {
                cause[0] = msg.data.haltinfo.mcause;
                cause[1] = msg.data.haltinfo.mcause2;
            }
        }
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
    {
        dprintf("_tdbgTaskGetSignal <- TDBG_MSG_ACK_FAIL(%"LwU64_fmtu")\n", msg.data.errcode.status);
        retVal = LW_ERR_ILWALID_STATE;
    }
    else
    {
        dprintf("_tdbgTaskGetSignal <- pktType incorrect\n");
        retVal = LW_ERR_ILWALID_STATE;
    }
    return retVal;
}

LwBool _tdbgTaskWaitForHalt(Session *pSession, LwU64 timeoutMs, LwS8 correctSignal, LwU64 *receivedSignal)
{
#if TASK_DEBUGGER_DEBUG_LEVEL > 1
    LwU64 origTimeoutMs = timeoutMs;
#endif
    LwBool bIsHalted = LW_FALSE;
    LwU64 delay = 1;
    LwBool isCorrectSignal = LW_FALSE;
    LwS64 cause[2] = {-1, -1};
    LwU64 *pCause = (LwU64 *) cause;

    if (receivedSignal != NULL) // Allow user to retrieve actual signal, or ignore with NULL
    {
        receivedSignal[0] = -1;
        receivedSignal[1] = -1;
        pCause = (LwU64 *) receivedSignal;
    }

    while (timeoutMs != 0)
    {
        riscvDelay((LwU32)delay);
        timeoutMs -= delay;
        delay = ((delay * 2) > timeoutMs) ? timeoutMs : delay * 2;
        delay = (delay > RISCV_RTOS_TIMEOUT_POLL_MS) ? RISCV_RTOS_TIMEOUT_POLL_MS : delay;
        if (_tdbgTaskGetSignal(pSession, &bIsHalted, pCause) != LW_OK) // failed to check if halted
        {
            break;
        }
        if (bIsHalted)
        {
            break;
        }
    }

    //
    // 0+: Standard RISCV mcause
    // -1: Task halted without exception
    // -2: Can't match, any signal is OK.
    //
    isCorrectSignal = (correctSignal == TDBG_SIGNAL_ANY) ? LW_TRUE : pCause[0] == correctSignal;

    if (bIsHalted && isCorrectSignal)
    {
#if TASK_DEBUGGER_DEBUG_LEVEL > 1
        dprintf("tdbgWaitForHalt: waited %"LwU64_fmtu" ms\n", origTimeoutMs - timeoutMs);
#endif
        return LW_TRUE;
    }
#if TASK_DEBUGGER_DEBUG_LEVEL > 1
    if (timeoutMs)
    {
        dprintf("tdbgWaitForHalt: wrong signal (got %"LwS64_fmtd" wanted %d), waited %"LwU64_fmtu" ms\n", pCause[0], (LwS32)correctSignal, origTimeoutMs - timeoutMs);
    }
    else
    {
        dprintf("tdbgWaitForHalt: timed out (%"LwU64_fmtu" ms)", origTimeoutMs);
    }
#endif
    return LW_FALSE;
}

LW_STATUS _tdbgTaskGo(Session *pSession)
{
    TaskDebuggerPacket cmd;
    cmd.header.pktType = TDBG_CMD_GO;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.ctrl);
    cmd.data.ctrl.target = pSession->rtosTask;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskAttach(Session *pSession, LwU64 *cause) // Target is supposed to be stopped after this call.
{
    LW_STATUS retVal = LW_ERR_GENERIC;
    TaskDebuggerPacket cmd, msg;

    cmd.header.pktType = TDBG_CMD_ATTACH;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.ctrl);
    cmd.data.ctrl.target = pSession->rtosTask;

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType == TDBG_MSG_ACK_OK) // Halted without signal. Done.
    {
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_HALTINFO) // Halted previously with signal.
    {
        cause[0] = msg.data.haltinfo.mcause;
        cause[1] = msg.data.haltinfo.mcause2;
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
    {
        dprintf("_tdbgTaskAttach <- TDBG_MSG_ACK_FAIL(%"LwU64_fmtu")\n", msg.data.errcode.status);
        retVal = LW_ERR_ILWALID_STATE;
    }
    else
    {
        dprintf("_tdbgTaskAttach <- pktType incorrect\n");
        retVal = LW_ERR_ILWALID_STATE;
    }
    return retVal;
}

LW_STATUS _tdbgTaskDetach(Session *pSession, LwU64 flags)
{
    TaskDebuggerPacket cmd;
    cmd.header.pktType = TDBG_CMD_DETACH;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.detach);
    cmd.data.detach.target = pSession->rtosTask;
    cmd.data.detach.flags = flags;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskClearStep(Session *pSession)
{
    TaskDebuggerPacket cmd;
    cmd.header.pktType = TDBG_CMD_CLEAR_STEP;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.ctrl);
    cmd.data.ctrl.target = pSession->rtosTask;
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

LW_STATUS _tdbgTaskStep(Session *pSession, const char *pRequest, LwU64 *cause)
{
    LW_STATUS retVal = LW_ERR_GENERIC;
    TaskDebuggerPacket cmd, msg;
    LwU64 waitTime = RISCV_RTOS_STEP_TICKS;
    LwBool bIsHalted;

    cmd.header.pktType = TDBG_CMD_USERSTEP;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.step);
    cmd.data.step.target = pSession->rtosTask;
    cmd.data.step.ticks = pSession->stepTicks;
    cause[0] = -1;
    cause[1] = -1;

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType == TDBG_MSG_ACK_OK) // Step completed
    {
        //
        // Implementation specific hack below.
        // Works: TU10x, GA100
        // Breaks: GA10x
        //
        cause[0] = 3; /* User mode single step */
        cause[1] = 25ULL << 56;
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_HALTINFO) // Step caused non-step exception
    {
        cause[0] = msg.data.haltinfo.mcause;
        cause[1] = msg.data.haltinfo.mcause2;
        dprintf("Step caused fault!\n");
        retVal = LW_OK;
    }
    else if (msg.header.pktType == TDBG_MSG_ACK_FAIL) // Step timed out
    {
        dprintf("_tdbgStep timed out on RTOS side. Task is likely waiting; polling forever!\n");
        do {
            dprintf("Waiting %"LwU64_fmtu" ms\n", waitTime);
            riscvDelay((LwU32)waitTime);
            waitTime = waitTime >= RISCV_RTOS_TIMEOUT_POLL_MS ? RISCV_RTOS_TIMEOUT_POLL_MS : waitTime * 2;

            cause[0] = -1;
            cause[1] = -1;
            retVal = _tdbgTaskGetSignal(pSession, &bIsHalted, cause);
            if (retVal != LW_OK)
            {
                dprintf("_tdbgStep: getSignal fail, aborting step attempt!\n");
                goto _tdbgStepAbort;
            }
        } while((cause[0] == -1) && (cause[1] == -1));
        dprintf("Step completed.\n");

        //
        // Normally when step works properly, the step flag is cleared before
        // the debugger task returns.
        //
        // When task stepping times out on RTOS side, the debugger task does not
        // clear the step flag. Assuming we're at this point in code, the
        // debuggee has resumed exelwtion and triggered an exception.
        // Therefore, we should clear the step flag now.
        //
        retVal = _tdbgTaskClearStep(pSession);
        if (retVal != LW_OK)
            dprintf("_tdbgStep: unable to clear step flag...\n");

        //
        // Implementation specific hack below.
        // Works: TU10x, GA100
        // Breaks: GA10x
        //
        if ((cause[0] == 3) && ((cause[1] >> 56) == 25)) // user mode single step
            retVal = LW_OK;
        else
        {
            dprintf("Step caused fault!\n");
            retVal = LW_OK;
        }
    }
    else
    {
        dprintf("_tdbgStep <- pktType incorrect\n");
        retVal = LW_ERR_ILWALID_STATE;
    }

_tdbgStepAbort:
    return retVal;
}

// Caller allocate.
LW_STATUS _tdbgTaskRegsRead(Session *pSession, LwU64 *xCtx)
{
    LW_STATUS retVal;
    TaskDebuggerPacket cmd;
    TaskDebuggerPacket msg;
    LwU64 remainingBytes;
    LwU64 bytesCopied;
    LwU8 *pCtx;

    cmd.header.pktType = TDBG_CMD_READ_TCB;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.tcbRW);
    cmd.data.tcbRW.target = pSession->rtosTask;
    cmd.data.tcbRW.size = RTOS_RISCV_CONTEXT_SIZE; /* TODO: floating point support */

    retVal = riscvSendCommand(pSession, &cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType != TDBG_MSG_BULKDATA_HDR)
    {
        if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
        {
            //User error (usually), or memory address bad (as in no access at all)
            dprintf("pktType == TDBG_MSG_ACK_FAIL, err=%08"LwU64_fmtx"\n", msg.data.errcode.status);
        }
        else
        {
            //Task Debugger protocol error
            dprintf("pktType != TDBG_MSG_BULKDATA_HDR\n");
        }
        return LW_ERR_ILWALID_STATE;
    }
    if (msg.data.bulkdata_hdr.type != TDBG_BULKDATA_TCB)
    {
        dprintf("Bulk Data hdr wrong type\n");
        return LW_ERR_ILWALID_STATE;
    }
    remainingBytes = msg.data.bulkdata_hdr.size;
    if (remainingBytes != RTOS_RISCV_CONTEXT_SIZE)
    {
        dprintf("Error, RTOS debugger returns invalid size of context %"LwU64_fmtu"\n", remainingBytes);
        // Any time we have to abort bulkdata transfer we need to reset debugger.
        tdbgResetDebugger(pSession);
        return LW_ERR_ILWALID_STATE;
    }
    // RTOS xTCB starts at x1
    pCtx = (LwU8 *) &xCtx[1];

    bytesCopied = 0;
    while (remainingBytes != 0)
    {
        cmd.header.pktType = TDBG_CMD_BULKDATA_ACK;
        cmd.header.dataSize = 0;

        retVal = riscvSendCommand(pSession, &cmd);
        if (retVal != LW_OK)
            return retVal;
        retVal = riscvRecvMessage(pSession, &msg);
        if (retVal != LW_OK)
            return retVal;

        if (msg.header.pktType != TDBG_MSG_BULKDATA)
        {
            dprintf("Unexpected end of BULKDATA transfer\n");
            tdbgResetDebugger(pSession);
            return LW_ERR_ILWALID_STATE;
        }
        memcpy(pCtx+bytesCopied, msg.data.raw.U8, msg.header.dataSize);
        bytesCopied += msg.header.dataSize;
        remainingBytes -= msg.header.dataSize;
    }
    xCtx[0] = 0; // x0 is always zero
    return LW_OK;
}

LW_STATUS _tdbgTaskRegsWrite(Session *pSession, LwU64 *xCtx, LwBool bSendPc)
{
    LW_STATUS retVal;
    TaskDebuggerPacket cmd;
    TaskDebuggerPacket msg;
    LwU32 remainingBytes;
    LwU32 bytesCopied;
    LwU32 packetSize;
    LwU8 *pCtx;

    pCtx = (LwU8 *) &xCtx[1]; // start sending from x1
    remainingBytes = bSendPc ? RTOS_RISCV_CONTEXT_SIZE : RTOS_RISCV_CONTEXT_SIZE - sizeof(LwU64);
    bytesCopied = 0;
    while (remainingBytes > 0)
    {
        memset(&cmd, 0, sizeof(cmd));
        packetSize = remainingBytes > sizeof(cmd.data.raw.U64) ? sizeof(cmd.data.raw.U64) : remainingBytes;

        cmd.header.pktType = TDBG_CMD_WRITE_LOCAL;
        cmd.header.dataSize = (LwU16) packetSize;
        memcpy(cmd.data.raw.U64, pCtx+bytesCopied, packetSize);

        bytesCopied += packetSize;
        remainingBytes -= packetSize;

        retVal = riscvSendCommand(pSession, &cmd);
        if (retVal != LW_OK)
            return retVal;
        memset(&msg, 0, sizeof(msg));
        retVal = riscvRecvMessage(pSession, &msg);
        if (retVal != LW_OK)
            return retVal;

        if (msg.header.pktType != TDBG_MSG_BYTESWRITTEN)
        {
            if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
            {
                //User error (usually), or memory address bad (as in no access at all)
                dprintf("pktType == TDBG_MSG_ACK_FAIL, err=%08"LwU64_fmtx"\n", msg.data.errcode.status);
            }
            else
            {
                //Task Debugger protocol error
                dprintf("pktType != TDBG_MSG_BYTESWRITTEN\n");
            }
            return LW_ERR_ILWALID_STATE;
        }

        if (msg.data.byteswritten.bytes != packetSize)
        {
            dprintf("Wrote less bytes (%"LwU64_fmtu") than sent (%lu). Probably out of RTOS memory.\n",
                    msg.data.byteswritten.bytes, (unsigned long) packetSize);
            // Ran out of debugger memory, DO NOT ATTEMPT TO WRITE TCB
            tdbgResetDebugger(pSession);
            return LW_ERR_ILWALID_STATE;
        }
    }

    cmd.header.pktType = TDBG_CMD_WRITE_TCB;
    cmd.header.dataSize = (LwU16) sizeof(cmd.data.tcbRW);
    cmd.data.tcbRW.target = pSession->rtosTask;
    cmd.data.tcbRW.size = bSendPc ? RTOS_RISCV_CONTEXT_SIZE : RTOS_RISCV_CONTEXT_SIZE - sizeof(LwU64); /* TODO: floating point support */
    return _tdbgCmdVerifyAck(pSession, &cmd);
}

// Helper Functions, declared
LW_STATUS _tdbgCmdVerifyAck(Session *pSession, TaskDebuggerPacket *cmd)
{
    LW_STATUS retVal;
    TaskDebuggerPacket msg;

    retVal = riscvSendCommand(pSession, cmd);
    if (retVal != LW_OK)
        return retVal;
    retVal = riscvRecvMessage(pSession, &msg);
    if (retVal != LW_OK)
        return retVal;

    if (msg.header.pktType == TDBG_MSG_ACK_OK)
        return LW_OK;
    else if (msg.header.pktType == TDBG_MSG_ACK_FAIL)
        dprintf(" <- TDBG_MSG_ACK_FAIL(%"LwU64_fmtu")\n", msg.data.errcode.status);
    else
        dprintf(" <- pktType incorrect\n");
    return LW_ERR_ILWALID_STATE;
}
