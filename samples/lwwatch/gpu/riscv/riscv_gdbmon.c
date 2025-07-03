/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <limits.h>

#include <os.h>
#include <utils/lwassert.h>

#include "riscv_gdbmon.h"
#include "riscv_prv.h"
#include "riscv_porting.h"

#include "hal.h"
#include "g_riscv_private.h"

typedef struct
{
    const char *pCommand;
    const char *pHelp;
    void (*pHandler)(void);
} MonitorCommand;

char monitorReplyBuf[MONITOR_BUF_SIZE] = {0,};
char *pMonitorSavedPtr = NULL; // for tokenizer

/**
 * 'Monitor' command handling
 */

static void _monitorDumpRstat(void)
{
    int ret, i;
    LwU64 val;
    char *pRbp = monitorReplyBuf;

    dprintf("Dumping rstat.\n");

    for (i=ICD_RSTAT0; i<=ICD_RSTAT_END; ++i)
    {
        ret = riscvIcdReadRstat(i, &val);
        if (ret == LW_OK)
            ret = sprintf(pRbp, "RSTAT(%d) = "LwU64_FMT"\n", i, val);
        else
            ret = sprintf(pRbp, "RSTAT(%d) = ?? Err: %x\n", i, ret);

        pRbp += ret;
    }
}

#define MK_BOOL(x) ((x) ? "Enabled" : "Disabled")
static void _monitorDumpConfig(void)
{
    char *pRbp = monitorReplyBuf;

    sprintf(pRbp,
            "Stub configuration: NONE\n"
            "Print GDB communication [printc]: %s\n"
            "    Include MEM transactions [printm]: %s\n"
            "    Include target communication [printc]: %s\n",
            MK_BOOL(config.bPrintGdbCommunication),
            MK_BOOL(config.bPrintMemTransactions),
            MK_BOOL(config.bPrintTargetCommunication));
}

static void _monitorSetConfig(void)
{
    int set = 0;
    char *pName;
    char *pRbp = monitorReplyBuf;

    pName = strtok_r(NULL, " \t", &pMonitorSavedPtr);
    if (!pName || pName[0] == 0)
    {
        sprintf(pRbp, "Usage: set [+|-]variable\n");
        return;
    }
    set = pName[0] == '-' ? 0 : 1; // clear only if requested
    if (pName[0] == '+' || pName[0] == '-')
        pName++;

    if (strcmp(pName, "printc") == 0)
    {
        sprintf(pRbp, "Setting bPrintGdbCommunication to %s\n", MK_BOOL(set));
        config.bPrintGdbCommunication = set;
    }
    else if (strcmp(pName, "printm") == 0)
    {
        sprintf(pRbp, "Setting bPrintMemTransactions to %s\n", MK_BOOL(set));
        config.bPrintMemTransactions = set;
    }
    else if (strcmp(pName, "printc") == 0)
    {
        sprintf(pRbp, "Setting bPrintTargetCommunication to %s\n", MK_BOOL(set));
        config.bPrintTargetCommunication = set;
    }
    else
    {
        sprintf(pRbp, "Unknown variable: %s\n", pName);
    }
}

static void _monitorSpin(void)
{
    riscvDelay(1);
    sprintf(monitorReplyBuf, "Spin completed.\n");
}

static void _monitorHelp(void);

// This is a hack, since the struct initializers below must be constant
static void _monitorReadCommsWrapper(void) {
    pRiscv[indexGpu]._monitorReadComms();
}

static void _monitorWriteHostWrapper(void) {
    pRiscv[indexGpu]._monitorWriteHost();
}

#define MK_CMD(name, fcn, help) { name, help, fcn }
static MonitorCommand cmds[] =
{
    MK_CMD("help", _monitorHelp, "Displays help\n"),
    MK_CMD("rstat", _monitorDumpRstat, "Dump RSTAT registers\n"),
    MK_CMD("config", _monitorDumpConfig, "Dump current configuration\n"),
    MK_CMD("set", _monitorSetConfig, "Set configuration variable\n"),
    MK_CMD("rc", _monitorReadCommsWrapper, "Read mtohost and mfromhost\n"),
    MK_CMD("wh", _monitorWriteHostWrapper, "Write mfromhost\n"),
    MK_CMD("spin", _monitorSpin, "Spin emulation\n"),
    MK_CMD(NULL, NULL, NULL)
};

static void _monitorHelp(void)
{
    MonitorCommand *pCmd = cmds;
    int ret;
    char *pRbp = monitorReplyBuf;

    while (pCmd->pCommand)
    {
        if ((strlen(pCmd->pCommand) + strlen(pCmd->pHelp) + 4) >
            (unsigned)(MONITOR_BUF_SIZE - (pRbp - monitorReplyBuf)))
        {
            dprintf("Ran out of space in monitorReplyBuf. Please expand it.\n");
            return;
        }
        ret = sprintf(pRbp, "%s - %s", pCmd->pCommand, pCmd->pHelp);
        pRbp += ret;

        pCmd++;
    }
}

LW_STATUS riscvGdbMonitor(char *pCmd, const char **ppReply)
{
    MonitorCommand *pCommand = cmds;
    const char *pToken;

    LW_ASSERT_OR_RETURN(pCmd != NULL, -1);
    LW_ASSERT_OR_RETURN(ppReply != NULL, -1);

    monitorReplyBuf[0] = 0;
    *ppReply = monitorReplyBuf;

    pMonitorSavedPtr = 0;
    pToken = strtok_r(pCmd, " \t", &pMonitorSavedPtr);

    if (pToken)
    {
        // Kill command - just return error
        if (strcmp("kill", pToken) == 0)
            return LW_ERR_RESET_REQUIRED;

        while (pCommand->pCommand)
        {
            if (strcmp(pToken, pCommand->pCommand) == 0)
            {
                pCommand->pHandler();
                return LW_OK;
            }
            pCommand++;
        }
    }

    _monitorHelp();

    // Always return success
    return LW_OK;
}
