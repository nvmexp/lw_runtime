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
#include <stdlib.h>
#include <lwtypes.h>
#include "lwstatus.h"

#include "riscv.h"
#include "riscv_prv.h"
#include "riscv_printing.h"

char gdbPath[512] = {0};
char gdbElf[512]  = {0};
char gdbCmds[512] = {0};

LW_STATUS gdbProcessElwVars() {
    if (*gdbPath == 0) {
        char *elwPath = getelw("LWWATCH_RISCV_GDBPATH");
        if (elwPath == NULL) {
            dprintf("Please set the environment variable `LWWATCH_RISCV_GDBPATH` to the location "
                    "of riscv64-elf-gdb.\n");
            return LW_ERR_ILWALID_STATE;
        }
        strncpy(gdbPath, elwPath, sizeof(gdbPath));
    }

    if (*gdbElf == 0) {
        char *elwPath = getelw("LWWATCH_RISCV_ELFPATH");
        if (elwPath == NULL) {
            dprintf("Warning: Target binary is not set. You can set it with the "
                    "environment variable `LWWATCH_RISCV_ELFPATH`\n");
        } else {
            strncpy(gdbElf, elwPath, sizeof(gdbElf));
        }
    }

    if (*gdbCmds == 0) {
        char *elwPath = getelw("LWWATCH_RISCV_GDBCMDS");
        if (elwPath != NULL) {
            strncpy(gdbCmds, elwPath, sizeof(gdbCmds));
        }
    }

    return LW_OK;
}

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>

static void *stubThreadRun(void *arg)
{
    LW_STATUS *ret = (LW_STATUS *) arg;
    debugPrints = LW_FALSE;
    LW_STATUS r = gdbStub(pRiscvInstance, RISCV_GDBSTUB_INIT_TIMEOUT_MS, LW_TRUE);
    debugPrints = LW_TRUE;
    if (r != LW_OK) {
        dprintf("gdb stub encountered an error\n");
        *ret = r;
        return ret;
    }
    *ret = LW_OK;
    return ret;
}

LW_STATUS gdbScriptRun(const char *script, enum GDB_MODE mode)
{
    int pid;

    if (gdbProcessElwVars() != LW_OK) {
        return LW_ERR_ILWALID_STATE;
    }

    if (riscvIsInactive()) {
        dprintf("Core must be started to enter debugger.\n");
        return LW_ERR_ILWALID_STATE;
    }

    char init[64];
    snprintf(init, sizeof(init), "target remote %s:%d", "localhost", pRiscvInstance->defaultPort);

    // create gdb process
    pid = fork();
    if (pid == -1) {
        dprintf("fork failed\n");
        return LW_ERR_GENERIC;
    }

    if (pid == 0) {
        // in child

        char cmd[1024];
        // execv gdb
        switch (mode) {
        case GDB_MODE_INTERACTIVE:
            if (*gdbCmds != 0) {
                execlp(gdbPath, gdbPath, gdbElf, "-q", "-x", gdbCmds, "-ex", init, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -q -x %s -ex %s", gdbPath, gdbElf, gdbCmds, init);
            } else {
                execlp(gdbPath, gdbPath, gdbElf, "-q", "-ex", init, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -q -ex %s", gdbPath, gdbElf, init);
            }
            break;
        case GDB_MODE_ONELINE:
            if (*gdbCmds != 0) {
                execlp(gdbPath, gdbPath, gdbElf, "-batch", "-q",
                       "-x", gdbCmds, "-ex", init, "-ex", script, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -batch -q -x %s -ex %s -ex %s",
                         gdbPath, gdbElf, gdbCmds, init, script);
            } else {
                execlp(gdbPath, gdbPath, gdbElf, "-batch", "-q", "-ex", init, "-ex", script, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -batch -q -ex %s -ex %s", gdbPath, gdbElf, init, script);
            }
            break;
        case GDB_MODE_SCRIPT:
            if (*gdbCmds != 0) {
                execlp(gdbPath, gdbPath, gdbElf, "-batch", "-q",
                       "-x", gdbCmds, "-ex", init, "-x", script, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -q -x %s -ex %s -x %s", gdbPath, gdbElf, gdbCmds, init, script);
            } else {
                execlp(gdbPath, gdbPath, gdbElf, "-batch", "-q", "-ex", init, "-x", script, (char *) NULL);
                snprintf(cmd, sizeof(cmd), "%s %s -q -ex %s -x %s", gdbPath, gdbElf, init, script);
            }
        }

        // if we get here, then execv failed
        dprintf("Could not start gdb: errno=%d\n", errno);
        dprintf("Command used: %s\n", cmd);
        exit(1);
    }
    else {
        // in parent

        // spin off thread for the stub
        pthread_t stubThread;
        LW_STATUS threadRetVal = LW_ERR_GENERIC;
        pthread_create(&stubThread, NULL, stubThreadRun, &threadRetVal);

        // wait for gdb child process
        int r;
        wait(&r);

        // wait for gdb stub thread
        if (pthread_join(stubThread, NULL)) {
            dprintf("error joining stub thread\n");
            return LW_ERR_GENERIC;
        }
        return threadRetVal;
    }
}

LW_STATUS gdbStubKill() {
    return gdbStub(NULL, 0, LW_FALSE);
}

#endif
