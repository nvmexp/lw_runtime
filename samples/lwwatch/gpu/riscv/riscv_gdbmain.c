/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "os.h"
#include "hal.h"
#include "exts.h"
#include "print.h"

#include "riscv.h"
#include "riscv_prv.h"
#include "riscv_config.h"
#include "riscv_dbgint.h"
#include "riscv_taskdbg.h"
#include "riscv_porting.h"
#include "lwsocket.h"
#include "riscv_cmd.h"

// for file operations
#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

//gdbScriptRun checks if core is inactive already, so we don't need to check.
static LW_STATUS cmdGdb(const char *pArgs)
{
    // run GDB interactive mode
    stubSetDebugPrints(LW_FALSE);
    return gdbScriptRun(NULL, GDB_MODE_INTERACTIVE);
}

static LW_STATUS cmdRemote(const char *pArgs)
{
    LwBool inactive;
    // NOTE: Lock is unlocked between this check and starting stub.
    // This is not a problem because the only thing that would cause problems here is a core reset,
    //   user is only one who can cause core reset, and user's lwwatch shell is hanging here.
    lwmutex_lock(icdLock);
    inactive = riscvIsInactive();
    lwmutex_unlock(icdLock);
    if (inactive)
    {
        dprintf("Core must be started to enter debugger.\n");
        return LW_ERR_ILWALID_STATE;
    }

    dprintf("Running GDB stub for remote debugging with a connection timeout of 60s.\n");
    stubSetDebugPrints(LW_TRUE);
    return gdbStub(pRiscvInstance, 60000, LW_FALSE);
}

static LW_STATUS cmdRun(const char *pArgs)
{
    stubSetDebugPrints(LW_FALSE);
    return gdbScriptRun(pArgs, GDB_MODE_ONELINE);
}

static LW_STATUS cmdScript(const char *pArgs)
{
    stubSetDebugPrints(LW_FALSE);
    return gdbScriptRun(pArgs, GDB_MODE_SCRIPT);
}

static LW_STATUS cmdKillStub(const char *pArgs)
{
    return gdbStubKill();
}

static LW_STATUS cmdHelp(const char *req);

#define MK_CMD(name, fcn, help) { name, help, fcn }
static Command commands[] =
{
    MK_CMD("interactive", cmdGdb, "\n\tStarts an interactive GDB session."),
    MK_CMD("kstub", cmdKillStub, "\n\tKills a GDB stub running in the background."),
    MK_CMD("run", cmdRun, "<cmd>\n\tRuns a one line command in GDB and prints the output to the console."),
    MK_CMD("remote", cmdRemote, "\n\tStarts the gdb stub for remote debugging."),
    MK_CMD("script", cmdScript, "<script>\n\tRuns a GDB script and prints the output to the console."),
    MK_CMD("where", cmdRun, "\n\tPrints the current location in the code. Equivalent to `rvgdb run` with no argument."),
    MK_CMD("help", cmdHelp, "Help."),
    { NULL, NULL }
};

static LW_STATUS cmdHelp(const char *req)
{
    Command *cmd = commands;

    dprintf("Setup: Use the following environment variables to configure rvgdb.\n" \
            "\tLWWATCH_RISCV_GDBPATH: location of RISC-V GDB\n" \
            "\tLWWATCH_RISCV_ELFPATH: location of target elf file\n" \
            "\tLWWATCH_RISCV_GDBCMDS: location of GDB script to run on GDB startup\n" \
            "Use your GDBCMDS file to define custom GDB commands and run any setup commands.\n" \
            "See https://confluence.lwpu.com/display/LW/GDB+Interface for more information.\n");
    dprintf("\nCommands:\n");
    while (cmd->command)
    {
        if (!req || (strlen(req) == 0) || (strcmp(req, cmd->command) == 0))
            dprintf("\n%s %s\n", cmd->command, cmd->help);
        cmd++;
    }
    return LW_OK;
}

static LwBool fileExists(const char *filename)
{
#if LWWATCHCFG_IS_PLATFORM(WINDOWS)

    return GetFileAttributesA(filename) != ILWALID_FILE_ATTRIBUTES;

#elif LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

    struct stat info;
    int r = stat(filename, &info);
    if (r < 0) return LW_FALSE;
    return LW_TRUE;

#endif
}

LW_STATUS riscvGdbMain(const char *pArgs)
{
    const char *pAllArgs = pArgs;
    const char *pCmd = NULL;
    int cmdLen = 0;
    Command *cmd = commands;

    LW_STATUS ret;

    if (!pRiscv[indexGpu].riscvIsSupported()) {
        dprintf("RISC-V is not supported on this device.\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    if (riscvInit() != LW_OK)
        return LW_ERR_GENERIC;

    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_GSP], RISCV_INSTANCE_GSP);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_SEC2], RISCV_INSTANCE_SEC2);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_PMU], RISCV_INSTANCE_PMU);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_MINION], RISCV_INSTANCE_MINION);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_TSEC], RISCV_INSTANCE_TSEC);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC0], RISCV_INSTANCE_LWDEC0);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC1], RISCV_INSTANCE_LWDEC1);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC2], RISCV_INSTANCE_LWDEC2);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC3], RISCV_INSTANCE_LWDEC3);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC4], RISCV_INSTANCE_LWDEC4);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC5], RISCV_INSTANCE_LWDEC5);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC6], RISCV_INSTANCE_LWDEC6);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_LWDEC7], RISCV_INSTANCE_LWDEC7);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_FSP], RISCV_INSTANCE_FSP);
    pRiscv[indexGpu].riscvPopulateCoreInfo(&instances[RISCV_INSTANCE_SOE], RISCV_INSTANCE_SOE);

    // preconfigure for now
    config.bPrintGdbCommunication = 1;
    config.bPrintMemTransactions = 1;
    config.bPrintTargetCommunication = 0;

    dprintf("Build %s %s on %s RISC-V. Args: %s\n", __DATE__, __TIME__, pRiscvInstance->name, pArgs);

    pArgs = riscvGetToken(pArgs, &pCmd, &cmdLen);

    // launch interactive mode on empty command
    if (!pCmd || cmdLen == 0 || *pCmd == 0)
        return cmdGdb(pAllArgs);

    // handle built-in commands
    while (cmd->command)
    {
        if (!strncmp(cmd->command, pCmd, cmdLen))
        {
            ret = cmd->handler(pArgs);
            if (ret != LW_OK)
                dprintf("Command returned error: 0x%x\n", ret);
            if (ret == LW_ERR_ILWALID_OPERATION)
                return cmdHelp(pCmd);
            return ret;
        }
        cmd++;
    }

    // if not a built-in command, see if it's a script
    if (fileExists(pCmd))
        return cmdScript(pAllArgs);

    // if not a built-in command or script, run as one-line GDB command
    return cmdRun(pAllArgs);

}
