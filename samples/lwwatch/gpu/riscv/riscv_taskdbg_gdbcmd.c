/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

// Includes
#include <utils/lwassert.h>

#include "lwsocket.h"
#include "riscv_gdbstub.h"
#include "riscv_prv.h"

#include "riscv_taskdbg_gdbcmd.h"
#include "riscv_gdbstub_common.h"
#include "riscv_taskdbg_communication.h"

static LW_STATUS _tdbgGdbSendSignal(Session *pSession, LwU64 *taskSignal)
{
    char retSignal[4];
    LwU32 trap;

    if (taskSignal[0] == -1) // Task halted because we stopped it.
    {
        // UNIX signal "SIGTRAP"
        strcpy(retSignal, "S05");
    }
    else // Stopped because of previous exception
    {
        trap = rvSIGINT;
        //
        // Breakpoints are handled differently.
        // Handle EBREAK as a SIGINT so stepping will not allow it to continue.
        // HW TRIGGER and UMSS should signal SIGTRAP so debugger can step multiple.
        //
        // Implementation specific hack below.
        // Works: TU10x, GA100
        // Breaks: GA10x
        //
        if (taskSignal[0] == 3)
        {
            trap = ((taskSignal[1]>>56) == 23) ? rvSIGINT : rvSIGTRAP;
        }

#if  LWWATCHCFG_IS_PLATFORM(WINDOWS)
        // reduce security for Windows
        sprintf(retSignal, "S%02u", (LwU32) trap);
#else
        snprintf(retSignal, 4, "S%02u", (LwU32) trap);
#endif
        dprintf("*** Returning signal: %s (original signal: %"LwU64_fmtx":%"LwU64_fmtx")\n", retSignal, taskSignal[0], taskSignal[1]);
    }
    return (rvGdbStubSendStr(pSession, retSignal) == -1) ? LW_ERR_GENERIC : LW_OK;
}

// GDB Stub Communication Functions
static int _tdbgGdbMemRead(Session *pSession, const char *pRequest)
{
    LwU64 addr = 0;
    unsigned len = 0;
    unsigned actual_len;
    LW_STATUS ret;

    if (sscanf(pRequest, "%"LwU64_fmtx",%x", &addr, &len) != 2)
        return rvGdbStubSendError(pSession, -1);

    if (!len)
        return rvGdbStubSendError(pSession, -1);
    else
    {
        LwU8 *pBuf = NULL;
        int rv;

        pBuf = malloc(len);

        if (!pBuf)
            return rvGdbStubSendError(pSession, -1);
        memset(pBuf, 0, len); // Buffer will be cleared by _tdbgTaskMemRead

        actual_len = len;
        ret = _tdbgTaskMemRead(pSession, pBuf, addr, &actual_len);
        if (ret == LW_OK)
        {
            // If (len != actual_len), it's not fatal error. GDB understands.
            rv = rvGdbStubSendPacket(pSession, pBuf, actual_len, GDB_SEND_HEX_ENCODED);
        }
        else
            rv = rvGdbStubSendError(pSession, ret);

        free(pBuf);
        return rv;
    }
}

static int _tdbgGdbMemWrite(Session *pSession, const char *pRequest)
{
    LwU64 addr = 0;
    LwU64 offset = 0;
    LwU8 *pBuf = NULL;
    int c;

    unsigned len = 0;
    unsigned actual_len;
    unsigned packet_len;
    TaskDebuggerPacket dummyPkt;
    LW_STATUS ret = LW_OK;

    if (sscanf(pRequest, "%"LwU64_fmtx",%x", &addr, &len) != 2)
        return rvGdbStubSendError(pSession, -1);

    if (!len)
        return rvGdbStubSendError(pSession, -1);

    pRequest = strstr(pRequest, ":");
    if (!pRequest)
        return rvGdbStubSendError(pSession, -1);

    pRequest++; // skip ":"
    if (strlen(pRequest) != len * 2)
        return rvGdbStubSendError(pSession, -1);

    if (len > 1)
    {
        int rv;

        pBuf = malloc(len);
        if (!pBuf)
            return rvGdbStubSendError(pSession, -1);

        memset(pBuf, 0, len);

        rvGdbStubHelperDecodeBE(pRequest, pBuf, len);

        offset = 0;
        while (len > 0)
        {
            packet_len = (len > sizeof(dummyPkt.data.memW.data64)) ? sizeof(dummyPkt.data.memW.data64) : len;
            actual_len = (LwU64) packet_len;
            ret = _tdbgTaskMemWritePacket(pSession, pBuf, addr+offset, &actual_len);
            if ((actual_len != packet_len) || (ret != LW_OK))
            {
                // Die if packet can't be fully written. This is error (according to GDB)
                ret = LW_ERR_NO_MEMORY;
                break;
            }
            offset += actual_len;
            len -= (unsigned)actual_len;
        }

        if (ret == LW_OK)
            rv = rvGdbStubSendOK(pSession);
        else
            rv = rvGdbStubSendError(pSession, ret);
        free(pBuf);
        return rv;
    }
    else
    {
        actual_len = 1;
        c = rvGdbStubHelperGetByte(pRequest);
        pBuf = (LwU8 *) &c; //this is not endian-safe
        ret = _tdbgTaskMemWritePacket(pSession, pBuf, addr, &actual_len);

        if (ret != LW_OK)
            return rvGdbStubSendError(pSession, ret);

        return rvGdbStubSendOK(pSession);
    }
}

static int _tdbgGdbRegsRead(Session *pSession)
{
    // When GDB does a register read, it does not request PC.
    // Since _tdbgTaskRegs functions are used to read PC, must overallocate here.
    LwU64 gdbCtx[(RISCV_GPR_COUNT + 1)] = {0, };
    char reply[RISCV_GPR_COUNT * 16 + 1] = {0, };

    LW_STATUS ret;
    int i;

    ret = _tdbgTaskRegsRead(pSession, gdbCtx);
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);

    for (i=0; i < RISCV_GPR_COUNT; i++)
    {
        rvGdbStubHelperEncodeBE(reply + i * 16, gdbCtx[i], 8);
        dprintf("Encoded x%d as %16s\n", i, reply + i * 16);
    }
    return rvGdbStubSendPacket(pSession, reply, sizeof(reply), 0);
}

static int _tdbgGdbRegRead(Session *pSession, const char *pRequest)
{
    // GDB uses this function to read PC.
    LW_STATUS ret;
    int regNo;
    LwU64 gdbCtx[(RISCV_GPR_COUNT + 1)] = {0, };
    char reply[16 + 1] = {0, };

    if (sscanf(pRequest, "%x", &regNo) != 1)
        return rvGdbStubSendError(pSession, -1);

    ret = _tdbgTaskRegsRead(pSession, gdbCtx);
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, -1);

    rvGdbStubHelperEncodeBE(reply, gdbCtx[regNo], 8);
    return rvGdbStubSendPacket(pSession, reply, sizeof(reply), 0);
}

static int _tdbgGdbRegsWrite(Session *pSession, const char *pRequest)
{
    // GDB never uses this function.
    LwU64 gdbCtx[(RISCV_GPR_COUNT + 1)] = {0, };
    LW_STATUS ret;
    int i;

    if (strlen(pRequest) < (16 * RISCV_GPR_COUNT))
    {
        dprintf("Truncated packet received.\n");
        return rvGdbStubSendError(pSession, -1);
    }

    for (i=0; i<RISCV_GPR_COUNT; ++i) // There is most certainly bug in original ICD debug stub.
    {
        rvGdbStubHelperDecodeBE(pRequest, &gdbCtx[i], 8);
        dprintf("write reg:%i="LwU64_FMT"\n", i, gdbCtx[i]);
        pRequest += 16;
    }
    ret = _tdbgTaskRegsWrite(pSession, gdbCtx, LW_FALSE); // Do not send PC.
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);
    return rvGdbStubSendOK(pSession);
}

static int _tdbgGdbRegWrite(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    int regNo;
    LwU64 gdbCtx[(RISCV_GPR_COUNT + 1)] = {0, };
    char *pEnd;

    regNo = strtoul(pRequest, &pEnd, 16);
    if (pEnd == pRequest || *pEnd == 0)
        return rvGdbStubSendError(pSession, -1);

    ret = _tdbgTaskRegsRead(pSession, gdbCtx);
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);

    pRequest = pEnd + 1; // skip '='
    rvGdbStubHelperDecodeBE(pRequest, &gdbCtx[regNo], 8);

    ret = _tdbgTaskRegsWrite(pSession, gdbCtx, LW_TRUE); // Send PC in case it's changed

    if (ret == LW_OK)
        return rvGdbStubSendOK(pSession);
    else if (ret == LW_ERR_NOT_SUPPORTED)
        return rvGdbStubSendEmpty(pSession);
    else
        return rvGdbStubSendError(pSession, ret);
}

////////////////////////////////////// COMMAND PARSING /////////////////////////

// Parse v-something commands
static int _tdbgStubV(Session *pSession, const char *pQuery)
{
    if (strcmp(pQuery, "Cont?") == 0)
        return rvGdbStubSendStr(pSession, "vCont;c;s;t");

    // vFile - LWSYM
    if (strncmp(pQuery, "File:", 5) == 0)
    {
        // TODO: implement once GSP has LWSYM support
        dprintf("Implement vFile once LWSYM support is ready.\n");
    }

    dprintf("Invalid V-packet received: '%s'.\n", pQuery);
    return rvGdbStubSendEmpty(pSession);
}

// Parse Q-something comands
static int _tdbgStubQuery(Session *pSession, const char *pQuery)
{
    dprintf("Query %p %s\n", pSession, pQuery);

    // thread ID,  big-endian hex string
    if (strcmp(pQuery, "C") == 0)
        return rvGdbStubSendStr(pSession, "00");

    // List of supported features
    if (strncmp(pQuery, "Supported", 9) == 0) // ignore what debugger supports
        return rvGdbStubSendStr(pSession, "hwbreak+;swbreak-;PacketSize=FF0");

    // Offsets
    if (strcmp(pQuery, "Offsets") == 0)
        return rvGdbStubSendStr(pSession, "Text=00000000;Data=00000000;Bss=00000000");

    // gdb can serve symbol lookup requests
    if (strcmp(pQuery, "Symbol::") == 0)
        return rvGdbStubSendOK(pSession);

    if (strncmp(pQuery, "Rcmd,", 5) == 0)
    { // "monitor" command
        char *pDecodedQuery = rvGdbStubAsciiDecode(pQuery + 5);
        const char *pReply = NULL;
        LW_STATUS ret;
        size_t replyLen;

        if (!pDecodedQuery)
        {
            rvGdbStubSendError(pSession, -1);
            return -1;
        }

        ret = riscvGdbMonitor(pDecodedQuery, &pReply);
        free(pDecodedQuery);

        if (ret != LW_OK)
        {
            if (ret == LW_ERR_RESET_REQUIRED)
                return REQUEST_DETACH;
            return rvGdbStubSendError(pSession, ret);
        }

        if (!pReply || strlen(pReply) == 0)
            return rvGdbStubSendOK(pSession);

        // Command generated some output - encode it, and send
        replyLen = strlen(pReply);
        return rvGdbStubSendPacket(pSession, pReply, replyLen, GDB_SEND_HEX_ENCODED);
    }

    dprintf("Invalid Query received: '%s'\n", pQuery);
    return rvGdbStubSendEmpty(pSession);
}

////////////////////////////////////// Command processing

// Translate stub return codes into sane return code
static REQUEST_RESULT _tdbgTrResult(int result)
{
    if (result < 0)
        return REQUEST_ERROR;
    return REQUEST_OK;
}

static REQUEST_RESULT _tdbgGdbProcessRequest(Session *pSession, GdbPacket *pRequest)
{
    LW_STATUS status;
    LwU64 detachFlags;
    LwU64 taskSignal[2] = {-1, -1};
    int ret = -1;

    LW_ASSERT_OR_RETURN(pSession, REQUEST_ERROR);
    LW_ASSERT_OR_RETURN(pRequest, REQUEST_ERROR);
    LW_ASSERT_OR_RETURN(pRequest->len > 0, REQUEST_ERROR);

    // We need to resume session - wait for CPU stop, then stop it forcefully
    if (pSession->state == SESSION_SUSPENDED)
    {
        dprintf("Waiting for target to halt.\n");

        if (_tdbgTaskAttach(pSession, taskSignal) != LW_OK)
            return REQUEST_ERROR;
        ret = _tdbgGdbSendSignal(pSession, taskSignal);
        return _tdbgTrResult(ret);
    }

    if (pRequest->data[0]!='m' || config.bPrintMemTransactions)
        dprintf("gdb request '%s'\n", pRequest->data);

    // Special treatment for "continue" - send confirmation and suspend debugger
    if (pRequest->data[0] == 'c')
    {
        dprintf("Continue requested.\n");
        if (_tdbgTaskGo(pSession) != LW_OK)
            return REQUEST_ERROR;
        return REQUEST_SUSPEND; // suspend (while waiting for interrupt)
    }

    switch (pRequest->data[0])
    {
        case '?': // Stop reason. Very similar code to resume.
            taskSignal[0] = -1;
            taskSignal[1] = -1;
            status = _tdbgTaskAttach(pSession, taskSignal);
            if (status == LW_OK)
                ret = _tdbgGdbSendSignal(pSession, taskSignal);
            else
                ret = -1;
            break;
        case 'g': // read general registers
            ret = _tdbgGdbRegsRead(pSession);
            break;
        case 'G': // write general registers
            ret = _tdbgGdbRegsWrite(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'H': // Set thread ID
            ret = rvGdbStubSendOK(pSession);
            break;
        case 'k': // kill
            dprintf("Detaching debugger\n");
            detachFlags = 0;
            if (pSession->exitAction == RESTORE_PREVIOUS)
            {
                detachFlags |= pSession->bTaskWasRunning ? TDBG_CMD_DETACH_RESUME : 0;
            }
            else
            {
                detachFlags |= (pSession->exitAction == ALWAYS_CONTINUE) ? TDBG_CMD_DETACH_RESUME : 0;
            }
            status = _tdbgTaskDetach(pSession, detachFlags);
            if (status != LW_OK)
            {
                rvGdbStubSendError(pSession, -1);
                break;
            }
            return REQUEST_DETACH;
        case 'm': // read memory
            ret = _tdbgGdbMemRead(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'M': // write memory
            ret = _tdbgGdbMemWrite(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'p': // read register "n" - BIG ENDIAN
            ret = _tdbgGdbRegRead(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'P': // write register "n"
            ret = _tdbgGdbRegWrite(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'q': // Extended query
        case 'Q':
            dprintf("%s:%d\n", __FUNCTION__, __LINE__);
            ret = _tdbgStubQuery(pSession, (const char*)&pRequest->data[1]);
            if (ret == REQUEST_DETACH)
                return ret;
            break;
        case 's': // step
            taskSignal[0] = -1;
            taskSignal[1] = -1;
            ret = _tdbgTaskStep(pSession, (const char*)&pRequest->data[1], taskSignal);
            if (ret != LW_OK)
                rvGdbStubSendError(pSession, ret);
            else
                _tdbgGdbSendSignal(pSession, taskSignal);
            break;
        case 'v': // v* request
            ret = _tdbgStubV(pSession, (const char*)&pRequest->data[1]);
            break;
        case 'X': // Read data - binary
            ret = rvGdbStubSendEmpty(pSession);
            break;
        case 'z': // remove breakpoint
            ret = _tdbgTaskClearBreakpoint(pSession, (const char*)&pRequest->data[1]);
            if (ret != LW_OK)
                rvGdbStubSendError(pSession, ret);
            else
                rvGdbStubSendOK(pSession);
            break;
        case 'Z': // insert breakpoint
            ret = _tdbgTaskSetBreakpoint(pSession, (const char*)&pRequest->data[1]);
            if (ret != LW_OK)
                rvGdbStubSendError(pSession, ret);
            else
                rvGdbStubSendOK(pSession);
            break;
        default:
            dprintf("Unknown request received: %5s...\n", pRequest->data);
            ret = rvGdbStubSendEmpty(pSession);
            break;
    }

    return _tdbgTrResult(ret);
}

LW_STATUS _taskDebuggerStubInstance(const RiscVInstance *pInstance, Session *pSession, LwU64 instanceId)
{
    LW_STATUS ret = LW_OK;

    if (pSession == NULL)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (pSession->rtosTask == 0)
    {
        dprintf("Fatal: xTCB address unset.\n");
        return LW_ERR_ILWALID_STATE;
    }

    // If no instance provided - destroy session and return;
    if (!pInstance)
    {
        if (pSession->state != SESSION_ILWALID)
        {
            dprintf("Destroying session for %s task 0x%"LwU64_fmtx"\n", pSession->pInstance->name, pSession->rtosTask);
            rvGdbStubDeleteSession(pSession);
        }
        return LW_OK;
    }

    // Instance switched
    if ((pSession->state != SESSION_ILWALID) && (pSession->pInstance != pInstance))
    {
        dprintf("Destroying session for %s task 0x%"LwU64_fmtx"\n", pSession->pInstance->name, pSession->rtosTask);
        rvGdbStubDeleteSession(pSession);
    }

    if (pSession->state == SESSION_ILWALID)
    {
        dprintf("Creating new session for %s task 0x%"LwU64_fmtx"\n", pInstance->name, pSession->rtosTask);
        dprintf("Task Debugger session info: xTCB addr 0x%16"LwU64_fmtx"\nContinue Delay (ms): %"LwU64_fmtu" %s\nTask status: %s\n",
                    pSession->rtosTask, pSession->continuePollMs, pSession->continuePollMs ? "" : "(disabled)",
                    ((pSession->exitAction == ALWAYS_CONTINUE) || (pSession->exitAction == ALWAYS_HALT)) ? "ignored" : "restored on exit"
                );
        CHECK_SUCCESS_OR_RETURN(rvGdbStubInitSession(pSession));
        pSession->pInstance = pInstance;
    }

    if (pSession->state == SESSION_CREATED)
    {// Wait for GDB
        dprintf("Preparing IP Server\n");
        // TODO: Figure out which port we can use.
        LW_ASSERT_OR_GOTO(lwSocketListen(pSession->pConnection, pInstance->defaultPort + (int)instanceId) == LW_OK, err);
        LW_ASSERT_OR_GOTO(lwSocketAccept(pSession->pConnection, -1) == LW_OK, err_server);
        pSession->state = SESSION_LISTENING;
    }

    // This means GDB is connected
    if (pSession->state == SESSION_LISTENING)
    pSession->state = SESSION_ATTACHED;

    dprintf("Attaching to target\n");

    // Finish suspended request first
    if (pSession->state == SESSION_SUSPENDED)
    {
        REQUEST_RESULT rq;
        dprintf("Resuming past session.\n");
        ret = LW_OK;
        rq = _tdbgGdbProcessRequest(pSession, pSession->pPacket);
        if (rq < 0)
        {
            ret = LW_ERR_GENERIC;
            goto err_client;
        }
        else if (ret == REQUEST_SUSPEND)
        {
            pSession->state = SESSION_SUSPENDED;
            goto out_keep;
        }
        else if (ret == REQUEST_DETACH)
        {
            goto err_client;
        }
        else
        {
            pSession->state = SESSION_ATTACHED;
        }
    }

    // Process messages in attached session
    while (pSession->state == SESSION_ATTACHED)
    {
        int rv;
        REQUEST_RESULT rq;

        ret = LW_OK;

        rv = rvGdbStubReceivePacket(pSession, pSession->pPacket);
        if (rv < 0)
        {
            dprintf("Failed reading packet.\n");
            pSession->state = SESSION_CLOSING;
            ret = LW_ERR_GENERIC;
            break;
        }

        rq = _tdbgGdbProcessRequest(pSession, pSession->pPacket);
        switch (rq)
        {
            case REQUEST_OK:
                continue;
            case REQUEST_ERROR:
                dprintf("Failed processing gdb request.\n");
                ret = LW_ERR_GENERIC;
                goto err_client;
            case REQUEST_DETACH:
                goto err_client;
            case REQUEST_SUSPEND:
                if (pSession->continuePollMs != 0)
                {
                    LwU64 cause[2] = {-1, -1};
                    if (_tdbgTaskWaitForHalt(pSession, pSession->continuePollMs, TDBG_SIGNAL_ANY, cause) == LW_TRUE)
                    {
                        ret = _tdbgGdbSendSignal(pSession, cause);
                        if (ret != LW_OK)
                            goto err_client;
                        continue;
                    }
                }
                pSession->state = SESSION_SUSPENDED;
                dprintf("Suspending GDB session.\n");
                goto out_keep;
            default:
                dprintf("Invalid value returned from request processor.\n");
                ret = LW_ERR_GENERIC;
                goto err_client;
        }
    }

err_client:
    lwSocketCloseClient(pSession->pConnection);
err_server:
    lwSocketCloseServer(pSession->pConnection);
err:
    rvGdbStubDeleteSession(pSession);

out_keep: // return while keeping session open
    return ret;
}
