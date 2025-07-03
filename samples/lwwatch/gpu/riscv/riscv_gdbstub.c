/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
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
#include "riscv_prv.h"
#include "riscv_porting.h"
#include "riscv_printing.h"

#include "riscv_gdbstub_common.h"

////////////////////////////////////// COMMAND PARSING /////////////////////////

// Parse v-something commands
static int _stubV(Session *pSession, const char *pQuery)
{
    if (strcmp(pQuery, "Cont?") == 0)
        return rvGdbStubSendStr(pSession, "vCont;c;s;t");

    // vFile - LWSYM
    if (strncmp(pQuery, "File:", 5) == 0)
    {
      // TODO: implement once GSP has LWSYM support
        lprintf(PL_DEBUG, "gdbstub: Implement vFile once LWSYM support is ready.\n");
    }

    lprintf(PL_DEBUG, "gdbstub: Invalid V-packet received: '%s'.\n", pQuery);
    return rvGdbStubSendEmpty(pSession);
}

// Parse Q-something comands
static int _stubQuery(Session *pSession, const char *pQuery)
{
    lprintf(PL_DEBUG, "Query %p %s\n", pSession, pQuery);

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
    lprintf(PL_DEBUG, "gdbstub: Invalid Query received: '%s'\n", pQuery);
    return rvGdbStubSendEmpty(pSession);
}

//Zn,299,2
// n == 0 : sw breakpoint
// n == 1 : hw breakpoint
// n == 2 : write watchpoint
// n == 3 : read watchpoint
// n == 4 : access watchpoint
static int _stubBreakSet(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    LwU64 addr;
    TRIGGER_EVENT flags = rvGdbStubMapBreakpointToTrigger(pRequest[0]);

    if (flags == TRIGGER_UNUSED)
        return rvGdbStubSendError(pSession, -1);

    pRequest = strstr(pRequest, ",");
    if (*pRequest)
        pRequest++;
    addr = strtoull(pRequest, NULL, 16);

    ret = pRiscv[indexGpu].riscvTriggerSetAt(addr, flags);
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);

    return rvGdbStubSendOK(pSession);
}

static int _stubBreakClear(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    LwU64 addr;
    TRIGGER_EVENT flags = rvGdbStubMapBreakpointToTrigger(pRequest[0]);

    if (flags == TRIGGER_UNUSED)
        return rvGdbStubSendError(pSession, -1);

    pRequest = strstr(pRequest, ",");
    if (*pRequest)
        pRequest++;
    addr = strtoull(pRequest, NULL, 16);

    ret = pRiscv[indexGpu].riscvTriggerClearAt(addr, flags);
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);

    return rvGdbStubSendOK(pSession);
}

static int _stubStep(Session *pSession, const char *pRequest)
{
    LW_STATUS ret = riscvIcdStep();
    if (ret != LW_OK)
        return rvGdbStubSendError(pSession, ret);

    return rvGdbStubSendStr(pSession, "S05"); // Fake "stop" command, return SIGTRAP
}

static int _stubRegsRead(Session *pSession)
{
    char reply[32 * 16 + 1] = {0, }; // 16 characters per register
    LW_STATUS ret;
    int i;

    memset(reply, 0, sizeof(reply));

    for (i=0; i<32; ++i)
    {
        LwU64 val;

        ret = pRiscv[indexGpu].riscvIcdRReg(i, &val);
        if (ret != LW_OK)
            return rvGdbStubSendError(pSession, ret);

        rvGdbStubHelperEncodeBE(reply + i * 16, val, 8);
        lprintf(PL_DEBUG, "Encoded x%d as %16s\n", i, reply + i * 16);
    }
    return rvGdbStubSendPacket(pSession, reply, sizeof(reply), 0);
}

static int _stubRegRead(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    int regNo;
    LwU64 val;

    if (sscanf(pRequest, "%x", &regNo) != 1)
        return rvGdbStubSendError(pSession, -1);

    ret = pRiscv[indexGpu].riscvRegReadGdb(regNo, &val);
    if (ret == LW_OK)
    {
        char reply[16 + 1] = {0, };

        rvGdbStubHelperEncodeBE(reply, val, 8);
        return rvGdbStubSendPacket(pSession, reply, sizeof(reply), 0);
    } else
    {
        return rvGdbStubSendError(pSession, -1);
    }
}

static int _stubRegWrite(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    int regNo;
    LwU64 regVal;
    char *pEnd;

    regNo = strtoul(pRequest, &pEnd, 16);
    if (pEnd == pRequest || *pEnd == 0 )
        return rvGdbStubSendError(pSession, -1);

    pRequest = pEnd + 1; // skip '='
    rvGdbStubHelperDecodeBE(pRequest, &regVal, 8);

    ret = pRiscv[indexGpu].riscvRegWriteGdb(regNo, regVal);
    if (ret == LW_OK)
        return rvGdbStubSendOK(pSession);
    else if (ret == LW_ERR_NOT_SUPPORTED)
        return rvGdbStubSendEmpty(pSession);
    else
        return rvGdbStubSendError(pSession, ret);
}

static int _stubRegsWrite(Session *pSession, const char *pRequest)
{
    LW_STATUS ret;
    int i;

    if (strlen(pRequest) < (16 * 32))
    {
        lprintf(PL_ERROR, "gdbstub: Truncated packet received.\n");
        return rvGdbStubSendError(pSession, -1);
    }

    for (i=1; i<32; ++i)
    {
        LwU64 val;

        rvGdbStubHelperDecodeBE(pRequest, &val, 8);
        lprintf(PL_DEBUG, "write reg:%i="LwU64_FMT"\n", i, val);

        pRequest += 16;
        ret = pRiscv[indexGpu].riscvIcdWReg(i, val);
        if (ret != LW_OK)
            return rvGdbStubSendError(pSession, ret);
    }
    return rvGdbStubSendOK(pSession);
}

static int _stubMemRead(Session *pSession, const char *pRequest)
{
    LwU64 addr = 0;
    unsigned len = 0;
    LW_STATUS ret;

    if (sscanf(pRequest, "%"LwU64_fmtx",%x", &addr, &len) != 2)
        return rvGdbStubSendError(pSession, -1);

    if (!len)
        return rvGdbStubSendError(pSession, -1);

    if (len > 1)
    {
        unsigned char *pBuf = NULL;
        int rv;

        pBuf = malloc(len);

        if (!pBuf)
            return rvGdbStubSendError(pSession, -1);
        memset(pBuf, 0, len);

        ret = riscvMemRead(addr, len, pBuf, MEM_FORCE_ICD_ACCESS);
        if (ret == LW_OK)
            rv = rvGdbStubSendPacket(pSession, pBuf, len, GDB_SEND_HEX_ENCODED);
        else
            rv = rvGdbStubSendError(pSession, ret);

        free(pBuf);
        return rv;
    }
    else
    {
        unsigned c;

        if (riscvMemRead(addr, 1, &c, MEM_FORCE_ICD_ACCESS) != LW_OK)
            return rvGdbStubSendError(pSession, -1);

        return rvGdbStubSendPacket(pSession, &c, 1, GDB_SEND_HEX_ENCODED);
    }
}

static int _stubMemWrite(Session *pSession, const char *pRequest)
{
    LwU64 addr = 0;
    unsigned len = 0;
    LW_STATUS ret = LW_OK;

    if (sscanf(pRequest, "%llx,%x", &addr, &len) != 2)
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
        unsigned char *pBuf = NULL;
        int rv;

        pBuf = malloc(len);
        if (!pBuf)
            return rvGdbStubSendError(pSession, -1);

        memset(pBuf, 0, len);

        rvGdbStubHelperDecodeBE(pRequest, pBuf, len);
        ret = riscvMemWrite(addr, len, pBuf, MEM_FORCE_ICD_ACCESS);
        if (ret == LW_OK)
            rv = rvGdbStubSendOK(pSession);
        else
            rv = rvGdbStubSendError(pSession, ret);
        free(pBuf);
        return rv;
    }
    else
    {
        int c;

        c = rvGdbStubHelperGetByte(pRequest);
        ret = riscvMemWrite(addr, 1, &c, MEM_FORCE_ICD_ACCESS);
        if (ret != LW_OK)
            return rvGdbStubSendError(pSession, ret);

        return rvGdbStubSendOK(pSession);
    }
}

////////////////////////////////////// Command processing

// Translate stub return codes into sane return code
static REQUEST_RESULT _trResult(int result)
{
    if (result < 0)
        return REQUEST_ERROR;
    return REQUEST_OK;
}

static REQUEST_RESULT _processGdbRequest(Session *pSession, GdbPacket *pRequest)
{
    LW_STATUS status;
    int ret = -1;

    LW_ASSERT_OR_RETURN(pSession, REQUEST_ERROR);
    LW_ASSERT_OR_RETURN(pRequest, REQUEST_ERROR);
    LW_ASSERT_OR_RETURN(pRequest->len > 0, REQUEST_ERROR);

    // We need to resume session - wait for CPU stop, then stop it forcefully
    if (pSession->state == SESSION_SUSPENDED)
    {
        lprintf(PL_ERROR, "gdbstub: Should not be in this state.\n");
        return REQUEST_ERROR;
    }

    if (pRequest->data[0]!='m' || config.bPrintMemTransactions)
        lprintf(PL_DEBUG, "gdb request '%s'\n", pRequest->data);

    lwmutex_lock(icdLock);

    // Special treatment for "continue" - send confirmation and wait for halt or interrupt
    if (pRequest->data[0] == 'c')
    {
        unsigned int delay = 1;
        unsigned char bufCtrlC[NETWORK_BUFFER_SIZE];

        lprintf(PL_DEBUG, "Continue requested.\n");

        if (riscvIcdRun() != LW_OK)
        {
            lwmutex_unlock(icdLock);
            return REQUEST_ERROR;
        }

        // wait for either ICD halt or Ctrl-C from GDB
        while (1)
        {

            if ( pRiscv[indexGpu].riscvIsInIcd())
                break;
            // do a non-blocking read to check for Ctrl-C
            ret = lwSocketRead(pSession->pConnection, bufCtrlC, NETWORK_BUFFER_SIZE, LW_FALSE);
            if (ret >= 0)
            {
                if (bufCtrlC[0] == 0x03)
                {
                    lprintf(PL_DEBUG, "gdbstub: got Ctl-C.\n");
                    lprintf(PL_DEBUG, "gdbstub: Forcefully stopping target.\n");
                    if (riscvIcdStop() != LW_OK) {
                        lwmutex_unlock(icdLock);
                        lprintf(PL_ERROR, "gdbstub: Failed to stop target.\n");
                        return REQUEST_DETACH;
                    }
                    break;
                }
            }

            // unlock the lock during delay to give main lwwatch a chance to `rv halt`
            lwmutex_unlock(icdLock);
            riscvDelay(delay);
            lwmutex_lock(icdLock);

            // exponential backoff
            delay *= 2;
            delay = (delay > RISCV_ICD_POLL_MAX_MS) ? RISCV_ICD_POLL_MAX_MS : delay;
        }
        rvGdbStubSendStr(pSession, "S05");
        lwmutex_unlock(icdLock);
        return REQUEST_OK;
    }

    switch (pRequest->data[0])
    {
    case '?': // Stop reason
            // TODO: implement non-stop mode later
        if (!pRiscv[indexGpu].riscvIsInIcd())
            status = riscvIcdStop();
        else
            status = LW_OK;
        if (status == LW_OK)
            ret = rvGdbStubSendStr(pSession, "S05");
        else
            ret = -1;
        break;
    case 'g': // read general registers
        ret = _stubRegsRead(pSession);
        break;
    case 'G': // write general registers
        ret = _stubRegsWrite(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'H': // Set thread ID
        ret = rvGdbStubSendOK(pSession);
        break;
    case 'k': // kill
        // Remove all breakpoints
        lprintf(PL_DEBUG, "gdbstub: Detaching debugger\n");
        status = riscvTriggerClearAll();
        if (status != LW_OK)
        {
            rvGdbStubSendError(pSession, -1);
            break;
        }
        if (((pSession->exitAction == RESTORE_PREVIOUS) && (!pSession->bTaskWasRunning)) ||
            (pSession->exitAction == ALWAYS_HALT))
        {
            lprintf(PL_DEBUG, "gdbstub: Core not resumed. Use rv go to resume.\n");
        }
        else
        {
            status = riscvIcdRun();
            if (status != LW_OK)
            {
                rvGdbStubSendError(pSession, -1);
                break;
            }
        }
        return REQUEST_DETACH;
    case 'm': // read memory
        ret = _stubMemRead(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'M': // write memory
        ret = _stubMemWrite(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'p': // read register "n" - BIG ENDIAN
        ret = _stubRegRead(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'P': // write register "n"
        ret = _stubRegWrite(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'q': // Extended query
    case 'Q':
            lprintf(PL_DEBUG, "gdbstub: %s:%d\n", __FUNCTION__, __LINE__);
        ret = _stubQuery(pSession, (const char*)&pRequest->data[1]);
        if (ret == REQUEST_DETACH) {
            lwmutex_unlock(icdLock);
            return ret;
        }
        break;
    case 's': // step
        ret = _stubStep(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'v': // v* request
        ret = _stubV(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'X': // Read data - binary
        ret = rvGdbStubSendEmpty(pSession);
        break;
    case 'z': // remove breakpoint
        ret = _stubBreakClear(pSession, (const char*)&pRequest->data[1]);
        break;
    case 'Z': // insert breakpoint
        ret = _stubBreakSet(pSession, (const char*)&pRequest->data[1]);
        break;
    default:
        lprintf(PL_ERROR, "gdbstub: Unknown request received: %5s...\n", pRequest->data);
        ret = rvGdbStubSendEmpty(pSession);
        break;
    }

    lwmutex_unlock(icdLock);

    return _trResult(ret);
}

int gdbHasPendingRequest(Session *pSession)
{
    LW_ASSERT_OR_RETURN(pSession != NULL, -1);
    LW_ASSERT_OR_RETURN(pSession->pConnection != NULL, -1);

    return lwSocketHasData(pSession->pConnection) > 0;
}

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

#include <signal.h>

void handleSIGINT(int signum) {
    lprintf(PL_DEBUG, "gdbstub: Ctrl-C pressed, ignoring.\n");
}

#endif // PLATFORM UNIX

static Session riscvSession = {SESSION_ILWALID, };

// Should only switch instances if there is no GDB stub running
LwBool stubCheckInstanceSwitch(RiscVInstance *pInstance) {
    return riscvSession.state == SESSION_ILWALID || pInstance == riscvSession.pInstance;
}

LW_STATUS gdbStub(const RiscVInstance *pInstance, int connectTimeout, LwBool installSignalHandler)
{
    LW_STATUS ret = LW_OK;
    static Session *pSession = &riscvSession;

    // If no instance provided - destroy session and return;

    if (!pInstance)
    {
        if (pSession->state != SESSION_ILWALID)
        {
            lprintf(PL_DEBUG, "gdbstub: Destroying session for %s.\n", pSession->pInstance->name);
            rvGdbStubDeleteSession(pSession);
        }
        return LW_OK;
    }

    // Instance switched
    if ((pSession->state != SESSION_ILWALID) && (pSession->pInstance != pInstance))
    {
        lprintf(PL_DEBUG, "gdbstub: Destroying session for %s.\n", pSession->pInstance->name);
        rvGdbStubDeleteSession(pSession);
    }

    if (pSession->state == SESSION_ILWALID)
    {
        lprintf(PL_DEBUG, "gdbstub: Creating new session for %s\n", pInstance->name);
        CHECK_SUCCESS_OR_RETURN(rvGdbStubInitSession(pSession));
        pSession->pInstance = pInstance;
        // For debugger scripting
        pSession->exitAction = RESTORE_PREVIOUS;
        lwmutex_lock(icdLock);
        pSession->bTaskWasRunning = !pRiscv[indexGpu].riscvIsInIcd();
        lwmutex_unlock(icdLock);
    }

    if (pSession->state == SESSION_CREATED)
    {// Wait for GDB
        lprintf(PL_DEBUG, "Preparing IP Server\n");
        LW_ASSERT_OR_GOTO(lwSocketListen(pSession->pConnection, pInstance->defaultPort) == LW_OK, err);
        if (lwSocketAccept(pSession->pConnection, connectTimeout) != LW_OK) {
            goto err_server;
        }
        pSession->state = SESSION_LISTENING;
    }

    // This means GDB is connected
    if (pSession->state == SESSION_LISTENING)
        pSession->state = SESSION_ATTACHED;

    lprintf(PL_DEBUG, "Attaching to target\n");

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    // Install sigint handler
    if (installSignalHandler == LW_TRUE)
        signal(SIGINT, handleSIGINT);
#endif // PLATFORM UNIX

    // Finish suspended request first
    if (pSession->state == SESSION_SUSPENDED)
    {
        REQUEST_RESULT rq;
        lprintf(PL_DEBUG, "Resuming past session.\n");
        ret = LW_OK;
        rq = _processGdbRequest(pSession, pSession->pPacket);
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
            lprintf(PL_ERROR, "gdbstub: Failed reading packet.\n");
            pSession->state = SESSION_CLOSING;
            ret = LW_ERR_GENERIC;
            break;
        }

        rq = _processGdbRequest(pSession, pSession->pPacket);
        switch (rq)
        {
            case REQUEST_OK:
                continue;
            case REQUEST_ERROR:
                lprintf(PL_ERROR, "gdbstub: Failed processing gdb request.\n");
                ret = LW_ERR_GENERIC;
                goto err_client;
            case REQUEST_DETACH:
                goto err_client;
            case REQUEST_SUSPEND:
                lprintf(PL_ERROR, "gdbstub: Should not have gotten into this state!\n\tSuspending session.\n");
                pSession->state = SESSION_SUSPENDED;
                goto out_keep;
            default:
                lprintf(PL_ERROR, "gdbstub: Invalid value returned from request processor.\n");
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

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    // Uninstall sigint handler
    if (installSignalHandler == LW_TRUE)
        signal(SIGINT, SIG_DFL);
#endif // PLATFORM UNIX
    return ret;
}
