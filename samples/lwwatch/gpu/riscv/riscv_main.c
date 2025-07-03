/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <lwtypes.h>

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
#include "lwsync_porting.h"
#include "riscv_printing.h"
#include "riscv_io_dio.h"

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
#include "rmlogging.h"
#endif

RiscVInstance instances[RISCV_INSTANCE_END];

RiscVInstance *pRiscvInstance = &instances[RISCV_INSTANCE_GSP];

static LwBool initialized = LW_FALSE;
static LwBool instanceInitialized[RISCV_INSTANCE_END];

const char * const riscvInstanceNames[RISCV_INSTANCE_END] =
{
    "GSP",
    "SEC2",
    "PMU",
    "MINION",
    "TSEC",
    "LWDEC0",
    "LWDEC1",
    "LWDEC2",
    "LWDEC3",
    "LWDEC4",
    "LWDEC5",
    "LWDEC6",
    "LWDEC7",
    "FSP",
    "SOE",
};

lwmutex_t icdLock;

static LW_STATUS riscvSwitchInstance(RiscvInstanceType type, LwBool bInit);

// Initializes the mutex icdLock. Should only be run once at lwwatch start (not per-command)
LW_STATUS riscvInit() {
    char* initEngine;
    int i;
    LwBool useDefaultEngine;

    if (initialized)
        return LW_OK;

    icdLock = lwmutex_create();
    if (icdLock == NULL) {
        dprintf("Error initializing mutex.\n");
        return LW_ERR_GENERIC;
    }

    for (i = 0; i < RISCV_INSTANCE_END; i++)
    {
        instanceInitialized[i] = LW_FALSE;
    }

    useDefaultEngine = LW_TRUE;
    initEngine = getelw("LWWATCH_RISCV_INITENGINE");
    if (initEngine != NULL) {
        for (i = 0; i < RISCV_INSTANCE_END; i++) {
            if (strcasecmp(initEngine, riscvInstanceNames[i]) == 0) {
                if (riscvSwitchInstance(i, LW_TRUE) == LW_OK)
                {
                    useDefaultEngine = LW_FALSE;
                }
                break;
            }
        }
        if (useDefaultEngine)
        {
            dprintf("Engine not supported: %s, defaulting to GSP\n", initEngine);
        }
    }

    if (useDefaultEngine)
    {
        riscvSwitchInstance(RISCV_INSTANCE_GSP, LW_TRUE);
    }

    initialized = LW_TRUE;
    return LW_OK;
}

LwBool riscvCoreIsGsp(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_GSP];
}

LwBool riscvCoreIsSec2(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_SEC2];
}

LwBool riscvCoreIsPmu(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_PMU];
}

LwBool riscvCoreIsMinion(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_MINION];
}

LwBool riscvCoreIsTsec(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_TSEC];
}

LwBool riscvCoreIsLwdec(LwU32 instId)
{
    LwU32 instToEngine = RISCV_INSTANCE_LWDEC0 + instId;
    if (instToEngine > RISCV_INSTANCE_LWDEC_LAST)
    {
        return LW_FALSE;
    }
    return pRiscvInstance == &instances[instToEngine];
}

LwBool riscvCoreIsFsp(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_FSP];
}

LwBool riscvCoreIsSoe(void)
{
    return pRiscvInstance == &instances[RISCV_INSTANCE_SOE];
}

// returns token to rest of arguments (or skip if there is no address)
static LwU64 _tokenizeAddress(const char *pArgs, const char **ppNext)
{
    const char *pTok = NULL;
    int tokLen = 0;

    if (ppNext)
        *ppNext = NULL;

    if (!pArgs)
        return 0;

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    // Just single token
    if (ppNext && (!pArgs || strlen(pArgs) == 0))
    {
        *ppNext = pTok;
        return 0;
    }

    // address and token
    if (ppNext)
        *ppNext = pArgs;

    if (!pTok || tokLen == 0)
        return 0;

    return strtoul(pTok, NULL, 0);
}

// Returns 0 if address was not parsed correctly
static LwU64 _parseAddress(const char *pArgs)
{
    return _tokenizeAddress(pArgs, NULL);
}

/*
 * Starts RISC-V core (in normal mode).
 * It will work only if core was not running before due to behaviour of
 * CPUCTL registers.
 * If core is already running - it either has to be reset, or one of other
 * commands should be used (halt, go, jump etc.)
 *
 * If address is given, it is used as boot address. If "fb" is given as address,
 * core starts at offset 0 of framebuffer.
 */
static LW_STATUS cmdBoot(const char *pArgs)
{
    if (pRiscv[indexGpu].riscvIsActive())
    {
        dprintf("CPU is already running.\n");
        return LW_ERR_ILWALID_STATE;
    }
    if (strcmp("fb", pArgs) == 0)
        return pRiscv[indexGpu].riscvBoot(pRiscvInstance->riscv_fb_start, BOOT_WITHOUT_ICD);
    return pRiscv[indexGpu].riscvBoot(_parseAddress(pArgs), BOOT_WITHOUT_ICD);
}

/*
 * Starts RISC-V core (in debug mode).
 * It will work only if core was not running before due to behaviour of
 * CPUCTL registers.
 * If core is already running - it either has to be reset, or one of other
 * commands should be used (halt, go, jump etc.)
 *
 * If address is given, it is used as boot address. If "fb" is given as address,
 * core starts at offset 0 of framebuffer.
 */
static LW_STATUS cmdBootDbg(const char *pArgs)
{
    if (pRiscv[indexGpu].riscvIsActive())
    {
        dprintf("CPU is already running.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (strcmp("fb", pArgs) == 0)
        return pRiscv[indexGpu].riscvBoot(pRiscvInstance->riscv_fb_start, BOOT_IN_ICD);

    return pRiscv[indexGpu].riscvBoot(_parseAddress(pArgs), BOOT_IN_ICD);
}

static LW_STATUS cmdReset(const char *pArgs)
{
    CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvReset(RESET_ENGINE));

    dprintf("Core reset.\n");
    return LW_OK;
}

static LW_STATUS cmdJump(const char *pArgs)
{
    LwU64 addr = _parseAddress(pArgs);

    if (riscvIsInactive())
    {
        dprintf("Core not started. Booting in debug mode.\n");
        return pRiscv[indexGpu].riscvBoot(addr, BOOT_IN_ICD);
    }

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Stopping core.\n");
        CHECK_SUCCESS_OR_RETURN(riscvIcdStop());
        CHECK_SUCCESS_OR_RETURN(riscvWaitForHalt(RISCV_ICD_TIMEOUT_MS));
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdJump(addr));

    dprintf("PC set to %"LwU64_fmtx"\n", addr);
    return LW_OK;
}

static LW_STATUS cmdGo(const char *pArgs)
{
    LwU64 addr = _parseAddress(pArgs);

    if (riscvIsInactive())
    {
        dprintf("Core not started. Just booting.\n");
        return pRiscv[indexGpu].riscvBoot(addr, BOOT_WITHOUT_ICD);
    }

    if (riscvIsRunning())
    {
        dprintf("Core is running. No need to go.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (pArgs && (strcmp(pArgs, "") == 0))
    {
        CHECK_SUCCESS_OR_RETURN(riscvIcdRun());
        dprintf("Core resumed at pc.\n");
    } else
    {
        CHECK_SUCCESS_OR_RETURN(riscvIcdJump(addr));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRun());
        dprintf("Core resumed at "LwU64_FMT".\n", addr);
    }

    return LW_OK;
}

static LW_STATUS cmdHalt(const char *pArgs)
{
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core already in ICD.\n");
        return LW_ERR_ILWALID_STATE;
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdStop());
    CHECK_SUCCESS_OR_RETURN(riscvWaitForHalt(RISCV_ICD_TIMEOUT_MS));

    return LW_OK;
}

static LW_STATUS cmdStep(const char *pArgs)
{
    unsigned long count = 0;
    LwU64 oldpc, pc;
    LwBool bAutoStep = LW_FALSE;
    LwBool bBrokenPC = LW_FALSE;
    LwU32 autoStepCount = 0;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to do steps.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (pArgs)
    {
        if (strstr(pArgs, "auto") != 0)
        {
            bAutoStep = LW_TRUE;
            count = 0;
        }
        else
        {
            count = strtoul(pArgs, NULL, 0);
            if (!count)
                count = 1;
        }
    }

    if (count > 1) // only show on multi-step
    {
        dprintf("Exelwting core step %ld times.\n", count);
    }

    // Always need old pc for non-blind stepping.
    if (pRiscv[indexGpu].riscvIcdRPc(&oldpc) != LW_OK)
    {
        dprintf("Unable to read pc!\n");
        bBrokenPC = LW_TRUE;
        if (bAutoStep)
            return LW_ERR_ILWALID_STATE; // cannot auto step if we can't read PC.
    }

    if (!bAutoStep)
    {
        unsigned long step = count;
        pc = oldpc;
        while (step --)
        {
            oldpc = pc;
            CHECK_SUCCESS_OR_GOTO(riscvIcdStep(), stepFail);

            // In case where HW is broken we should allow blind stepping.
            if (!bBrokenPC)
                CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvIcdRPc(&pc));
        }
    }
    else
    {
        pc = oldpc;
        do {
            oldpc = pc;
            CHECK_SUCCESS_OR_GOTO(riscvIcdStep(), stepFail);
            CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvIcdRPc(&pc));
            autoStepCount++;
            if (autoStepCount == RISCV_AUTO_STEP_MAX)
                break;
        } while ((pc == (oldpc + 2)) || (pc == (oldpc + 4)));

        dprintf("Auto-stepped %u instruction%s%s\n", autoStepCount, autoStepCount > 1 ? "s" : "",
            autoStepCount == RISCV_AUTO_STEP_MAX ? " (limit hit)" : "");
    }

    if (pRiscv[indexGpu].riscvIcdRPc(&pc) == LW_OK)
    {
        // This heuristic is not bulletproof (partially because of HW bug), but it's better than nothing.
        dprintf("PC  = %16"LwU64_fmtx"%s\n", pc,
            (count!=1) || ((pc == (oldpc + 2)) || (pc == (oldpc + 4))) ? "" : " *");
    }
    else
    {
        dprintf("Unable to read pc, this should not happen.\n");
        return LW_ERR_ILWALID_STATE;
    }

    return LW_OK;

stepFail:
    if (bBrokenPC)
        dprintf("Step failed. PC is broken\n");
    else
        dprintf("Step failed. Last known PC\nPC  = %16"LwU64_fmtx"\n", oldpc);
    return LW_ERR_ILWALID_STATE;
}

static LW_STATUS cmdSpin(const char *pArgs)
{
    unsigned long delay = 10;

    if (pArgs)
        delay = strtoul(pArgs, NULL, 0);

    if (delay == 0)
        delay = 10;

    dprintf("Spinning fmodel(%lu)...\n", delay);
    riscvDelay(delay);
    return LW_OK;
}

static LW_STATUS cmdDbg(const char *pArgs)
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
    dprintf("Use `rvgdb i` to start an interactive GDB session.\n");
    stubSetDebugPrints(LW_TRUE);
    return gdbStub(pRiscvInstance, 60000, LW_FALSE);
}

static LW_STATUS cmdKillDbg(const char *pArgs)
{
    return gdbStubKill();
}

// Args: offset <name>
// ArgsAlt: <name>
static LW_STATUS _loadFile(const char *fname, LwU64 addr, enum MEM_TYPE dest)
{
    LW_STATUS ret = LW_OK;
    long size;
    FILE *fd;
    char buf[RISCV_MAX_UPLOAD];

    if (riscvIsRunning())
    {
        dprintf("Core is running. Can't program.\n");
        return LW_ERR_ILWALID_STATE;
    }

    fd = fopen(fname, "rb");
    if (!fd)
    {
        dprintf("Failed opening image file: '%s'\n", fname);
        return LW_ERR_OBJECT_NOT_FOUND;
    }

    fseek(fd, 0L, SEEK_END);
    size = ftell(fd);
    rewind(fd);



    while (size > 0)
    {
        long chunk = min(RISCV_MAX_UPLOAD , size);

        dprintf("Reading %ld bytes from file.\n", chunk);
        if (fread(buf, chunk, 1, fd) != 1)
        {
            if (feof(fd))
                dprintf("File seems empty.\n");
            else if (ferror(fd))
                dprintf("Error: %d\n", ferror(fd));
            else
                dprintf("Error reading file.\n");
            ret = LW_ERR_ILWALID_READ;
            goto out_fd;
        }

        // For inactive core use fast programming, at that point (for now)
        // we can only program imem/dmem
        if (riscvIsInactive() && dest != MEM_VIRT && dest != MEM_FB)
        {
            dprintf("Writing %ld bytes at 0x%"LwU64_fmtx" using PMB.\n", chunk, addr);
            if (dest == MEM_IMEM || dest == MEM_IMEMS)
                ret = pRiscv[indexGpu].riscvImemWrite((LwU32)addr, chunk, buf, backdoor_no, dest == MEM_IMEMS);
            else if (dest == MEM_DMEM)
                ret = pRiscv[indexGpu].riscvDmemWrite((LwU32)addr, chunk, buf, backdoor_no);
            else if (dest == MEM_EMEM)
                ret = pRiscv[indexGpu].riscvEmemWrite((LwU32)addr, chunk, buf, backdoor_no);
            else
                return LW_ERR_ILWALID_ARGUMENT;
        }
        else
        {
            // Active core should be in ICD
            if (pRiscv[indexGpu].riscvIsInIcd())
            {
                if (pRiscv[indexGpu].riscvHasMpuEnabled()) {
                    dprintf("Warning: MPU is enabled. Memory writes may go to unexpected locations.\n");
                }
                dprintf("Writing %ld bytes at 0x%"LwU64_fmtx"\n", chunk, addr);
                //
                // If memory type is FB - address is offset from start of FB
                //
                if (dest == MEM_FB)
                    addr = addr + pRiscvInstance->riscv_fb_start;
                if (dest == MEM_IMEM)
                    addr = addr + pRiscvInstance->riscv_imem_start;
                if (dest == MEM_DMEM)
                    addr = addr + pRiscvInstance->riscv_dmem_start;
                if (dest == MEM_EMEM)
                    addr = addr + pRiscvInstance->riscv_emem_start;

                // We have loadi/loadd/loade for PMB accelerated writes. Need one for slow ICD method with MPU disabled.
                if (dest == MEM_VIRT || dest == MEM_FB)
                    ret = riscvMemWrite(addr, chunk, buf, MEM_FORCE_ICD_ACCESS);
                else
                    ret = riscvMemWrite(addr, chunk, buf, MEM_SMART_ACCESS);
            }
            else
            {
                dprintf("Core must be Inactive or in ICD for writes.\n");
                ret = LW_ERR_GENERIC;
            }
        }

        if (ret != LW_OK)
            break;

        size -= chunk;
        addr += chunk;
    }
out_fd:
    fclose(fd);
    return ret;
}

static LW_STATUS cmdLoadi(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_IMEM);
}

static LW_STATUS cmdLoadis(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_IMEMS);
}

static LW_STATUS cmdLoadd(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_DMEM);
}

static LW_STATUS cmdLoade(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_EMEM);
}

static LW_STATUS cmdLoadfb(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_FB);
}

static LW_STATUS cmdLoad(const char *pArgs)
{
    LwU64 addr;

    addr = _tokenizeAddress(pArgs, &pArgs);

    if (!pArgs || strlen(pArgs) == 0)
        return LW_ERR_ILWALID_OPERATION;

    return _loadFile(pArgs, addr, MEM_VIRT);
}

static LW_STATUS _dumpMem(const char *pArgs, enum MEM_TYPE type)
{
    unsigned size;
    LwU64 addr;
    const char *pTok = NULL;
    int tokLen = 0;
    LW_STATUS ret = LW_OK;

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    addr = strtoull(pTok, NULL, 0);

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    size = strtoul(pTok, NULL, 0);

    if (size == 0)
    {
        dprintf("Error: Nothing to read.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (size > RISCV_MAX_UPLOAD)
    {
        dprintf("Error: Can't read more than %d bytes.\n", RISCV_MAX_UPLOAD);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    {
        char *pBuf = malloc(size);

        if (!pBuf)
            return LW_ERR_NO_MEMORY;

        memset(pBuf, 0, size);

        switch (type)
        {
        case MEM_IMEM:
        case MEM_IMEMS:
            ret = pRiscv[indexGpu].riscvImemRead(LwU64_LO32(addr), size, pBuf, backdoor_no);
            break;
        case MEM_DMEM:
            ret = pRiscv[indexGpu].riscvDmemRead(LwU64_LO32(addr), size, pBuf, backdoor_no);
            break;
        case MEM_EMEM:
            ret = pRiscv[indexGpu].riscvEmemRead(LwU64_LO32(addr), size, pBuf, backdoor_no);
            break;
        case MEM_VIRT:
            ret = riscvMemRead(addr, size, pBuf, MEM_FORCE_ICD_ACCESS);
            break;
        case MEM_FB:
            ret = riscvMemRead(pRiscvInstance->riscv_fb_start + addr, size, pBuf, MEM_FORCE_ICD_ACCESS);
            break;
        }

        if (ret == LW_OK)
            riscvDumpHex(pBuf, size, addr);

        free(pBuf);
    }
    return ret;
}

static LW_STATUS cmdDumpi(const char *pArgs)
{
    return _dumpMem(pArgs, MEM_IMEM);
}

static LW_STATUS cmdDumpd(const char *pArgs)
{
    return _dumpMem(pArgs, MEM_DMEM);
}

static LW_STATUS cmdDumpe(const char *pArgs)
{
    return _dumpMem(pArgs, MEM_EMEM);
}

static LW_STATUS cmdDumpVa(const char *pArgs)
{
    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to access Virtual Memory.\n");
        return LW_ERR_ILWALID_STATE;
    }

    return _dumpMem(pArgs, MEM_VIRT);
}

static LW_STATUS cmdDumpFb(const char *pArgs)
{
    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to access Virtual Memory.\n");
        return LW_ERR_ILWALID_STATE;
    }

    return _dumpMem(pArgs, MEM_FB);
}

//
// This command is Windows-only because it uses WinDBG and Windows OCA
//
#if LWWATCHCFG_IS_PLATFORM(WINDOWS) && !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
static LW_STATUS cmdLoadCore(const char *pArgs)
{
    LW_STATUS status = LW_OK;
    LwU32 bufferSize, actualSize;
    LwU8 *buffer;
    const char *pTok;
    int tokLen;
    const char *pFname;
    FILE *pFd;
    size_t wrote;

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
    {
        return LW_ERR_ILWALID_OPERATION;
    }

    if (!pArgs || strlen(pArgs) == 0)
    {
        bufferSize = RISCV_DEFAULT_COREDUMP_SIZE;
        pFname = pTok;
    }
    else
    {
        bufferSize = strtoul(pTok, NULL, 0);
        pFname = pArgs;
    }

    if (bufferSize == 0)
    {
        dprintf("Error: Nothing to read.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    buffer = malloc(bufferSize * sizeof(LwU8));

    if (buffer == NULL)
    {
        dprintf("Error: Could not allocate memory for the buffer.\n");
    }

    if (lwMode == MODE_LIVE)
    {
        status = getRiscvLiveCoreDump(buffer, bufferSize, &actualSize);
    }
    else if (lwMode == MODE_DUMP)
    {
        status = getRiscvCoreDumpFromProtobuf(buffer, bufferSize, &actualSize);
    }

    if (status != LW_OK)
    {
        goto exit;
    }

    dprintf("Loaded core dump.\n");

    if ((buffer[0] != 0x7f) || (buffer[1] != 'E') || (buffer[2] != 'L') || (buffer[3] != 'F'))
    {
        dprintf("Warning: Not an ELF file.\n");
    }

    // TODO: run gdb on core dump
    // For now, we save core dump as a file so it can be opened with GDB

    pFd = fopen(pFname, "w");
    if (pFd == NULL)
    {
        dprintf("Failed opening core file: \"%s\"\n", pFname);
        status = LW_ERR_OBJECT_NOT_FOUND;
        goto exit;
    }

    wrote = fwrite(buffer, actualSize, 1, pFd);
    fclose(pFd);

    if (wrote != 1)
    {
        dprintf("Error writing to file.\n");
        status = LW_ERR_ILWALID_OPERATION;
        goto exit;
    }

    dprintf("Wrote %d B to file \"%s\".\n", actualSize, pFname);

exit:
    free(buffer);
    return status;
}
#endif //LWWATCHCFG_IS_PLATFORM(WINDOWS)

static LW_STATUS cmdDumpCore(const char *pArgs)
{
    const char *pTok;
    int tokLen;
    LwU64 addr;
    unsigned size;
    const char *pFname;
    char buf[RISCV_MAX_UPLOAD];
    FILE *pFd;
    size_t wrote;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to access Virtual Memory.\n");
        return LW_ERR_ILWALID_STATE;
    }

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
    {
        return LW_ERR_ILWALID_OPERATION;
    }

    addr = strtoul(pTok, NULL, 0);

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
    {
        return LW_ERR_ILWALID_OPERATION;
    }

    if (!pArgs || strlen(pArgs) == 0)
    {
        // default to 1 page
        size = 0x1000;
        pFname = pTok;
    }
    else
    {
        size = strtoul(pTok, NULL, 0);
        pFname = pArgs;
    }

    if (size == 0)
    {
        dprintf("Error: Nothing to read.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (size > RISCV_MAX_UPLOAD)
    {
        dprintf("Error: Can't read more than %d bytes.\n", RISCV_MAX_UPLOAD);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    CHECK_SUCCESS_OR_RETURN(riscvMemRead(addr, size, buf, MEM_FORCE_ICD_ACCESS));
    if ((buf[0] != 0x7f) || (buf[1] != 'E') || (buf[2] != 'L') || (buf[3] != 'F'))
    {
        dprintf("Warning: Not an ELF file.\n");
    }

    pFd = fopen(pFname, "w");
    if (pFd == NULL)
    {
        dprintf("Failed opening core file: \"%s\"\n", pFname);
        return LW_ERR_OBJECT_NOT_FOUND;
    }

    wrote = fwrite(buf, size, 1, pFd);
    fclose(pFd);

    if (wrote != 1)
    {
        dprintf("Error writing to file.\n");
        return LW_ERR_ILWALID_OPERATION;
    }

    dprintf("Dumped core to: \"%s\"\n", pFname);
    return LW_OK;
}

/*
 * dmesg dumps the buffer using PMB
 * dmesg f flushes the buffer
 */
static LW_STATUS cmdDmesg(const char *req)
{
    if (req && ((strcmp("", req) == 0) || (strcmp("f", req) == 0)))
    {
        LwBool bFlush = req && (strcmp("f", req) == 0);
        return riscvDumpDmesg(bFlush, DMESG_VIA_PMB);
    }
    else
    {
        LwU64 addr = _parseAddress(req);
        if (addr <= (pRiscvInstance->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr)))
        {
            pRiscvInstance->riscv_dmesg_hdr_addr = addr;
            dprintf("DMESG buffer set to DMEM offset 0x%08"LwU64_fmtx"\n", pRiscvInstance->riscv_dmesg_hdr_addr);
        }
        else
        {
            dprintf("DMESG DMEM offset invalid: (0x%08"LwU64_fmtx" > 0x%08"LwU64_fmtx")\n", addr, (pRiscvInstance->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr)));
        }
        return LW_OK;
    }
}

/*
 * dmesge dumps the buffer using ICD. It should be used if buffer is placed
 * somewhere else and just mapped to the end of DMEM.
 * dmesge f flushes the buffer
 */
static LW_STATUS cmdDmesge(const char *req)
{
    LwBool bFlush = req && (strcmp("f", req) == 0);
    LwBool wasHalted = LW_FALSE;
    LW_STATUS ret;
    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is running, halting first..\n");
        CHECK_SUCCESS_OR_RETURN(riscvIcdStop());
        CHECK_SUCCESS_OR_RETURN(riscvWaitForHalt(1000));
        wasHalted = LW_TRUE;
    }
    ret = riscvDumpDmesg(bFlush, DMESG_VIA_ICD);

    if (wasHalted)
    {
        dprintf("Core was halted, resuming.\n");
        CHECK_SUCCESS_OR_RETURN(riscvIcdRun());
    }
    return ret;
}

static LW_STATUS cmdReadCsr(const char *pArgs)
{
    unsigned long addr;
    const char *pTok;
    char *pEnd;
    int tokLen;
    LwU64 csr;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to access CSR.\n");
        return LW_ERR_ILWALID_STATE;
    }

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    // Start with number, truncate to 16bit
    addr = strtoul(pTok, &pEnd, 0);

    // Invalid number, try to resolve symbolic address
    if (pEnd == pTok)
    {
        LwS16 ta;
        ta = pRiscv[indexGpu].riscvDecodeCsr(pTok, tokLen);
        if (ta < 0)
        {
            dprintf("Invalid CSR name / offset.\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }
        addr = ta;
    }

    if (addr > 0xFFF)
    {
        dprintf("Last CSR address is 0xFFF\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvIcdRcsr((LwU16)addr, &csr));

    dprintf("CSR[%s] = "LwU64_FMT"\n", pTok, csr);

    return LW_OK;
}

static LW_STATUS cmdWriteCsr(const char *pArgs)
{
    unsigned long addr;
    const char *pCsrName, *pValue;
    char *pEnd;
    int tokLen;
    LwU64 csr;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core must be in ICD to access CSR.\n");
        return LW_ERR_ILWALID_STATE;
    }

    pArgs = riscvGetToken(pArgs, &pCsrName, &tokLen);

    if (!pCsrName || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    // Start with number, truncate to 16bit
    addr = strtoul(pCsrName, &pEnd, 0);

    // Invalid number, try to resolve symbolic address
    if (pEnd == pCsrName)
    {
        LwS16 ta;

        ta = pRiscv[indexGpu].riscvDecodeCsr(pCsrName, tokLen);
        if (ta < 0)
        {
            dprintf("Invalid CSR name / offset.\n");
            return LW_ERR_ILWALID_ARGUMENT;
        }
        addr = ta;
    }

    if (addr > 0xFFF)
    {
        dprintf("Last CSR address is 0xFFF\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    // Now get the value
    riscvGetToken(pArgs, &pValue, &tokLen);
    if (!pValue || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    csr = strtoull(pValue, NULL, 0);

    dprintf("CSR[%lx] <= "LwU64_FMT"\n", addr, csr);

    CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvIcdWcsr((LwU16)addr, csr));

    return LW_OK;
}

static LW_STATUS cmdState(const char *pArgs)
{
    LW_STATUS ret;
    LwU64 rstat;
    const char *rstat4_icd_states[]={"ACTIVE", "INACTIVE", "ICD"};

    pRiscv[indexGpu].riscvDumpState();

    pRiscv[indexGpu].riscvIcdDumpRegs();

    ret = riscvIcdReadRstat(4, &rstat);
    if (ret != LW_OK)
        return ret;

    if ((rstat & 0x3) == 3)
    {
        dprintf("rstat4 = "LwU64_FMT" (illegal)", rstat);
    }
    else
    {
        dprintf("rstat4 = "LwU64_FMT" %s", rstat, rstat4_icd_states[rstat & 0x3]);
        if ((rstat & 0x7f0) != 0x7f0)
            dprintf(" TRIGGER_HIT[%"LwU64_fmtu"]", (rstat >> 4) & 0x7f);
    }
    dprintf("\n");

    return LW_OK;
}

static LW_STATUS cmdReload(const char *pArgs)
{
    CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvReset(RESET_ENGINE));
    CHECK_SUCCESS_OR_RETURN(_loadFile(pArgs, 0, MEM_IMEM));
    return pRiscv[indexGpu].riscvBoot(0, BOOT_IN_ICD);
}

/*!
 * @brief  Switch to a given RISCV instance. If the instance is not initialized, it'll be initialized first.
 * @param[in]  type  The RISCV instance to switch to.
 * @param[in]  bInit  Whether the function is called during plugin initialization.
 * @return     LW_OK on success.
 * @return     LW_ERR_NOT_SUPPORTED if instance not supported or not safe to swtich to.
 */
static LW_STATUS riscvSwitchInstance(RiscvInstanceType type, LwBool bInit)
{
    RiscVInstance *pInstance = &instances[type];
    if (!pRiscv[indexGpu].riscvIsInstanceSupported(type))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    // For threading.
    // Check and see if we have any debugger sessions open.
    if (!(taskDebuggerCheckInstanceSwitch(pInstance) && stubCheckInstanceSwitch(pInstance))) {
        dprintf("Not safe to switch instances: Please close all debugger sessions.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    pRiscvInstance =  pInstance;
    if (!instanceInitialized[type])
    {
        pRiscv[indexGpu].riscvPopulateCoreInfo(pRiscvInstance, type);
        instanceInitialized[type] = LW_TRUE;
    }
    if (!bInit)
    {
        dprintf("Switching to %s\n", pRiscvInstance->name);
    }

    return LW_OK;
}

static LW_STATUS cmdSwitchSec2(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_SEC2, LW_FALSE);
}

static LW_STATUS cmdSwitchGsp(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_GSP, LW_FALSE);
}

static LW_STATUS cmdSwitchPmu(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_PMU, LW_FALSE);
}

static LW_STATUS cmdSwitchMinion(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_MINION, LW_FALSE);
}

static LW_STATUS cmdSwitchTsec(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_TSEC, LW_FALSE);
}

static LW_STATUS cmdSwitchLwdec(const LwU32 instId, const char *pArgs)
{
    LwU32 instToEngine = RISCV_INSTANCE_LWDEC0 + instId;
    if (instToEngine > RISCV_INSTANCE_LWDEC_LAST)
    {
        return LW_ERR_NOT_SUPPORTED;
    } 
    return riscvSwitchInstance(instToEngine, LW_FALSE);
}

#define DEFINE_CMD_SWITCH_LWDEC(x)                           \
static LW_STATUS cmdSwitchLwdec##x(const char *pArgs) \
{                                                     \
    return cmdSwitchLwdec(x, pArgs);                  \
}

DEFINE_CMD_SWITCH_LWDEC(0)
DEFINE_CMD_SWITCH_LWDEC(1)
DEFINE_CMD_SWITCH_LWDEC(2)
DEFINE_CMD_SWITCH_LWDEC(3)
DEFINE_CMD_SWITCH_LWDEC(4)
DEFINE_CMD_SWITCH_LWDEC(5)
DEFINE_CMD_SWITCH_LWDEC(6)
DEFINE_CMD_SWITCH_LWDEC(7)

static LW_STATUS cmdSwitchFsp(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_FSP, LW_FALSE);
}

static LW_STATUS cmdSwitchSoe(const char *pArgs)
{
    return riscvSwitchInstance(RISCV_INSTANCE_SOE, LW_FALSE);
}

/*
 * MPU dumping routine
 */
static LW_STATUS cmdDumpMpu(const char *pArgs)
{
    LwU64 regions = _parseAddress(pArgs);

    return pRiscv[indexGpu].riscvDumpMpu(LwU64_LO32(regions));
}

static LW_STATUS cmdDumpPmp(const char *pArgs)
{
    LwU64 regions = _parseAddress(pArgs);

    return pRiscv[indexGpu].riscvDumpPmp(LwU64_LO32(regions));
}

static LW_STATUS cmdDumpIoPmp(const char *pArgs)
{
    LwU64 vaOffset = _parseAddress(pArgs);
    if (vaOffset)
    {
        dprintf("Setting PRIV VA base to 0x"LwU64_FMT"\n", vaOffset);
    }
    return pRiscv[indexGpu].riscvDumpIoPmp(vaOffset);
}

static LW_STATUS cmdDumpBreakpoint(const char *pArgs)
{
    LW_STATUS retVal = LW_ERR_GENERIC;
    const char *pCh = NULL;

    // Dump Breakpoint
    LwU64 regions;

    // Set Breakpoint
    LwU64 addr;
    LwU64 flags;

    // Set+Clear Breakpoint
    LwU64 index;

    if (!pArgs)
        return LW_ERR_ILWALID_OPERATION;

    switch(pArgs[0])
    {
        case 's': // set
            pCh = strstr(pArgs," ");
            if (pCh)
            {
                index = _parseAddress(pCh+1);
            }
            else
                return LW_ERR_ILWALID_ARGUMENT;

            pCh = strstr(pCh+1," ");
            if (pCh)
            {
                addr = _parseAddress(pCh+1);
            }
            else
                return LW_ERR_ILWALID_ARGUMENT;

            pCh = strstr(pCh+1," ");
            if (pCh)
            {
                flags = _parseAddress(pCh+1);
            }
            else
            {
                flags = pRiscv[indexGpu].riscvDefaultBpFlags();
            }

            retVal = pRiscv[indexGpu].riscvSetBreakpoint((int)index, addr, flags);
            break;
        case 'c': // clear
            pCh = strstr(pArgs," ");
            if (pCh)
            {
                if (!sscanf(pCh+1, "%"LwU64_fmtu"", &index))
                    return LW_ERR_ILWALID_ARGUMENT;
            }
            else
                return LW_ERR_ILWALID_ARGUMENT;

            retVal = pRiscv[indexGpu].riscvClearBreakpoint((int)index);
            break;
        case 'C':
            retVal = riscvClearAllBreakpoints();
            break;
        case 0:
        default:
            regions = _parseAddress(pArgs);
            retVal = pRiscv[indexGpu].riscvDumpBreakpoint(LwU64_LO32(regions));
            break;
    }

    return retVal;
}

static LW_STATUS cmdTrace(const char *pArgs)
{
    if (!pArgs)
        return LW_ERR_ILWALID_OPERATION;

    switch(pArgs[0])
    {
    case '0':
        dprintf("Enabling trace buffer in mode 0.\n");
        return pRiscv[indexGpu].riscvTraceEnable(0);
    case '1':
        dprintf("Enabling trace buffer in mode 1.\n");
        return pRiscv[indexGpu].riscvTraceEnable(1);
    case '2':
        dprintf("Enabling trace buffer in mode 2.\n");
        return pRiscv[indexGpu].riscvTraceEnable(2);
    case 'f':
        dprintf("Flushing trace buffer.\n");
        return pRiscv[indexGpu].riscvTraceFlush();
    case '-':
        dprintf("Disabling trace buffer.\n");
        return pRiscv[indexGpu].riscvTraceDisable();
    case 0:
        dprintf("Dumping trace buffer.\n");
        return pRiscv[indexGpu].riscvTraceDump();
    default:
        return LW_ERR_ILWALID_OPERATION;
    }
    return LW_OK;
}

/*
 * Helper function to access words in RISC-V address space.
 * - pArgs is address provided via cmdline
 * - base  is base address of subsystem that we want to access, to avoid
 *         writing (rather long) 64-bit address.
 */
static LW_STATUS _readInt(const char *pArgs, LwU64 base)
{
    LwU32 val;
    LwU64 addr;
    const char *pTok = NULL;
    int tokLen = 0;
    LW_STATUS ret = LW_OK;

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    addr = strtoull(pTok, NULL, 0);

    ret = riscvIcdRdm(addr + base, &val, ICD_WIDTH_32);

    dprintf("[%"LwU64_fmtx"] = %x\n", addr + base, val);
    return ret;
}
/*
 * Helper function to access words in RISC-V address space.
 * - pArgs are address and value provided via cmdline
 * - base  is base address of subsystem that we want to access, to avoid
 *         writing (rather long) 64-bit address.
 */
static LW_STATUS _writeInt(const char *pArgs, LwU64 base)
{
    LwU32 val;
    LwU64 addr;
    const char *pTok = NULL;
    int tokLen = 0;
    LW_STATUS ret = LW_OK;

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    addr = strtoull(pTok, NULL, 0);

    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);

    if (!pTok || tokLen == 0)
        return LW_ERR_ILWALID_OPERATION;

    val = strtoul(pTok, NULL, 0);

    dprintf("[%"LwU64_fmtx"] = %x\n", addr + base, val);
    ret = riscvIcdWdm(addr + base, val, ICD_WIDTH_32);
    return ret;
}

static LW_STATUS cmdRead(const char *pArgs)
{
    return _readInt(pArgs, 0);
}

static LW_STATUS cmdReadFb(const char *pArgs)
{
    return _readInt(pArgs, pRiscvInstance->riscv_fb_start);
}

static LW_STATUS cmdReadBar0(const char *pArgs)
{
    return _readInt(pArgs, pRiscvInstance->riscv_priv_start);
}

static LW_STATUS cmdWrite(const char *pArgs)
{
    return _writeInt(pArgs, 0);
}

static LW_STATUS cmdWriteFb(const char *pArgs)
{
    return _writeInt(pArgs, pRiscvInstance->riscv_fb_start);
}

static LW_STATUS cmdWriteBar0(const char *pArgs)
{
    return _writeInt(pArgs, pRiscvInstance->riscv_priv_start);
}

// cmd rv rdio se <addr>
// cmd rv wdio se <addr> <value>
// cmd rv rdio snic <addr>
// cmd rv wdio snic <addr> <value>
// cmd rv rdio extra <port> <addr>
// cmd rv wdio extra <port> <addr> <value>
static LW_STATUS _riscvDioReadWrite(const char *pArgs, DIO_OPERATION dioOp)
{
    const char *pTok = NULL;
    int tokLen = 0;

    DIO_PORT dioPort;
    LwU32 addr;
    LwU32 value = 0;

    LW_STATUS status = LW_OK;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }

    // get DIO type and DIO port index
    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
    if (strncmp("se", pTok, tokLen) == 0)
    {
        dioPort.dioType = DIO_TYPE_SE;
    }
    else if (strncmp("snic", pTok, tokLen) == 0)
    {
        dioPort.dioType = DIO_TYPE_SNIC;
        dioPort.portIdx = DIO_TYPE_SNIC_PORT_IDX;
    }
    else if (strncmp("extra", pTok, tokLen) == 0)
    {
        dioPort.dioType = DIO_TYPE_EXTRA;
        pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
        if (!pTok || tokLen == 0)
        {
            return LW_ERR_ILWALID_OPERATION;
        }
        dioPort.portIdx = strtoul(pTok, NULL, 0);
    }
    else
    {
        return LW_ERR_ILWALID_OPERATION;
    }

    // get address
    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
    if (!pTok || tokLen == 0)
    {
        return LW_ERR_ILWALID_OPERATION;
    }
    addr = strtoul(pTok, NULL, 0);

    // get data for write operation
    if (dioOp == DIO_OPERATION_WR)
    {
        pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
        if (!pTok || tokLen == 0)
        {
            return LW_ERR_ILWALID_OPERATION;
        }
        value = strtoul(pTok, NULL, 0);
    }

    status = riscvDioReadWrite(dioPort, dioOp, addr, &value);

    //
    // Print result, examples:
    // DIO_TYPE_SE    [0x12312] => 0x12312
    // DIO_TYPE_EXTRA [0x12312] <= 0x12312
    //
    dprintf("%s [0x%x] %s 0x%x\n", DIO_TYPE_STR[dioPort.dioType], addr, (dioOp == DIO_OPERATION_RD) ? "=>" : "<=", value);

    return status;
}

static LW_STATUS cmdDioRead(const char *pArgs)
{
    return _riscvDioReadWrite(pArgs, DIO_OPERATION_RD);
}

static LW_STATUS cmdDioWrite(const char *pArgs)
{
    return _riscvDioReadWrite(pArgs, DIO_OPERATION_WR);
}

// Task level debugger

static LW_STATUS cmdTaskDbgList(const char *pArgs)
{
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }
    return tdbgListTasks(pRiscvInstance);
}

static LW_STATUS cmdTaskDbgSessions(const char *pArgs)
{
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }
    return tdbgListSessions();
}

static LW_STATUS cmdTaskDbg(const char *pArgs)
{
    const char *pCh = pArgs;
    LwU64 xTCB = 0;
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }
    xTCB = _parseAddress(pCh);
    if (!xTCB)
    {
        dprintf("You must specify xTCB address for task debugger.\n");
        return LW_ERR_ILWALID_OPERATION;
    }
    dprintf("Starting task debugger...\n");
    return taskDebuggerStub(pRiscvInstance, xTCB, LW_FALSE, 0); // base impl
}

static LW_STATUS cmdTaskDbg2(const char *pArgs)
{
    const char *pCh = pArgs;
    LwU64 xTCB = 0;
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }
    xTCB = _parseAddress(pCh);
    if (!xTCB)
    {
        dprintf("You must specify xTCB address for task debugger.\n");
        return LW_ERR_ILWALID_OPERATION;
    }
    dprintf("Starting task debugger2...\n");
    return taskDebuggerStub(pRiscvInstance, xTCB, LW_FALSE, 10000); // base impl
}

static LW_STATUS cmdTaskDbg3(const char *pArgs)
{
    const char *pCh = pArgs;
    LwU64 xTCB = 0;
    if (pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Core is halted; task debugger requires running RTOS.\n");
        return LW_ERR_ILWALID_STATE;
    }
    xTCB = _parseAddress(pCh);
    if (!xTCB)
    {
        dprintf("You must specify xTCB address for task debugger.\n");
        return LW_ERR_ILWALID_OPERATION;
    }
    dprintf("Starting task debugger3...\n");
    return taskDebuggerStub(pRiscvInstance, xTCB, LW_TRUE, 10000); // base impl
}

static LW_STATUS cmdShowCoreSelwrity(const char *pArgs)
{
    return pRiscv[indexGpu].riscvGetLsInfo();
}


static LW_STATUS cmdBrStatus(const char *pArgs)
{
    return pRiscv[indexGpu].riscvBrStatus();
}

static long _fileSize(const char *fname)
{
    FILE *fd;
    long size;

    fd = fopen(fname, "rb");
    if (!fd)
    {
        dprintf("Failed opening file: '%s'\n", fname);
        return -1;
    }

    fseek(fd, 0L, SEEK_END);
    size = ftell(fd);
    fclose(fd);

    return size;
}

static LW_STATUS cmdBrBoot(const char *pArgs)
{
    char *textFileName = NULL;
    char *dataFileName = NULL;
    char *manifestFileName = NULL;
    LwBool bFreeText = LW_TRUE;
    LwBool bFreeData = LW_TRUE;
    LwBool bFreeManifest = LW_TRUE;
    const char *pTok;
    int tokLen;
    LW_STATUS ret = LW_OK;
    long manifestSize;


    pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
    if (tokLen)
    {
        textFileName = calloc(tokLen + 1, 1);
        strncpy(textFileName, pTok, tokLen);

        pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
        if (tokLen)
        {
            dataFileName = calloc(tokLen + 1, 1);
            strncpy(dataFileName, pTok, tokLen);

            pArgs = riscvGetToken(pArgs, &pTok, &tokLen);
            if (tokLen)
            {
                manifestFileName = calloc(tokLen + 1, 1);
                strncpy(manifestFileName, pTok, tokLen);
            }
        }
    }

    if (textFileName == NULL)
    {
        textFileName = "testapp-gsp.text.encrypt.bin";
        bFreeText = LW_FALSE;
    }
    if (dataFileName == NULL)
    {
        dataFileName = "testapp-gsp.data.encrypt.bin";
        bFreeData = LW_FALSE;
    }
    if (manifestFileName == NULL)
    {
        manifestFileName = "testapp-gsp.manifest.encrypt.bin.out.bin";
        bFreeManifest = LW_FALSE;
    }

    if (textFileName == NULL || dataFileName == NULL || manifestFileName == NULL)
    {
        // This should never happen.. seriously..
        ret = LW_ERR_INSUFFICIENT_RESOURCES;
        goto out;
    }

    dprintf("Resetting core *AND* SE...\n");
    CHECK_SUCCESS_OR_RETURN(pRiscv[indexGpu].riscvBrReset(LW_TRUE));

    dprintf("Loading images...\n");
    dprintf(".text image: %s\n", textFileName);
    CHECK_SUCCESS_OR_GOTO(_loadFile(textFileName, 0x0, MEM_IMEM), out);
    dprintf(".data image: %s\n", dataFileName);
    CHECK_SUCCESS_OR_GOTO(_loadFile(dataFileName, 0x0, MEM_DMEM), out);
    dprintf("manifest image: %s\n", manifestFileName);
    manifestSize = _fileSize(manifestFileName);
    if (manifestSize <= 0)
    {
        goto out;
    }
    // Manifest is loaded to end of dmem
    CHECK_SUCCESS_OR_GOTO(_loadFile(manifestFileName, pRiscvInstance->riscv_dmem_size - manifestSize, MEM_DMEM), out);

    dprintf("Starting core...\n");
    CHECK_SUCCESS_OR_GOTO(pRiscv[indexGpu].riscvBrBoot(LW_TRUE), out);

    dprintf("Core started... status:\n");
    cmdBrStatus("");
out:
    if (bFreeText)
    {
        free(textFileName);
    }
    if (bFreeData)
    {
        free(dataFileName);
    }
    if (bFreeManifest)
    {
        free(manifestFileName);
    }
    return ret;
}

static LW_STATUS cmdBrReset(const char *pArgs)
{
    return pRiscv[indexGpu].riscvBrReset(LW_TRUE);
}

static LW_STATUS cmdHelp(const char *req);

#define MK_CMD(name, fcn, lock, help) { name, help, fcn, lock }
static Command commands[]=
{
    MK_CMD("r", cmdRead, LW_TRUE, "<addr>\n\tReads word from RISC-V address space."),
    MK_CMD("w", cmdWrite, LW_TRUE, "<addr> <value>\n\tWrites word to RISC-V address space."),
    MK_CMD("rf", cmdReadFb, LW_TRUE, "<offset>\n\tReads word from framebuffer with RISC-V."),
    MK_CMD("wf", cmdWriteFb, LW_TRUE, "<offset> <value>\n\tWrites word to framebuffer with RISC-V."),
    MK_CMD("rb", cmdReadBar0, LW_TRUE, "<offset>\n\tReads word from bar0 with RISC-V."),
    MK_CMD("wb", cmdWriteBar0, LW_TRUE, "<offset> <value>\n\tWrites word to bar0 with RISC-V."),
    MK_CMD("rdio", cmdDioRead, LW_TRUE, "[se|snic <portIdx>] <offset>\n\tReads word from DIO with RISC-V."),
    MK_CMD("wdio", cmdDioWrite, LW_TRUE, "[se|snic <portIdx>] <offset> <value>\n\tWrites word through DIO with RISC-V."),
    MK_CMD("boot", cmdBoot, LW_TRUE, "[addr|fb]\n\tStarts RISC-V core. "
                            "If no address is given, starts at 0. If 'fb' is "
                            "given - boots at offset 0 of framebuffer."),
    MK_CMD("bootd", cmdBootDbg, LW_TRUE, "[addr|fb]\n\tStarts RISC-V core in dbg mode. "
                                "If no address is given, starts at 0. If 'fb' is "
                                "given - boots at offset 0 of framebuffer."),
    MK_CMD("brs", cmdBrStatus, LW_TRUE, "Prints bootroom status."),
    MK_CMD("brb", cmdBrBoot, LW_TRUE, "[imem-file] [dmem-file] [manifest-file]\n\tBoots riscv in bootrom "
                                "mode (image names hardcoded as testapp-<engine>.<data|text>.encrypt.bin, "
                                "testapp-<engine>.manifest.encrypt.bin.out.bin). "
                                "Filenames CANT have spaces."),
    MK_CMD("brr", cmdBrReset, LW_TRUE, "Resets RISC-V *and* SE."),
    MK_CMD("reset", cmdReset, LW_TRUE, "\n\tResets RISC-V (and Falcon) core."),
    MK_CMD("jump", cmdJump, LW_TRUE, "<address>\n\tEnters debug mode and sets PC to a given address."),
    MK_CMD("go", cmdGo, LW_TRUE, "[address]\n\tResumes RISC-V core (leaves debug mode). "
                        "If address is given - resumes at address."),
    MK_CMD("halt", cmdHalt, LW_TRUE, "\n\tStops RISC-V core (enters debug mode)."),
    MK_CMD("step", cmdStep, LW_TRUE, "[num]/'auto'\n\tDoes single ICD step [num] times. 'auto' steps until !sequential exelwtion."),
    MK_CMD("spin", cmdSpin, LW_FALSE, "\n\tShort delay (spin for fmodel)."),
    MK_CMD("dbg", cmdDbg, LW_FALSE, "Starts GDB stub for remote debugging. Consider using `rvgdb` extension instead."),
    MK_CMD("gdb", cmdDbg, LW_FALSE, "Starts GDB stub for remote debugging. Consider using `rvgdb` extension instead."),
    MK_CMD("kdbg", cmdKillDbg, LW_FALSE, "\n\tKills an existing GDB stub instance. May be needed if gdb or stub have misbehaved."),
    MK_CMD("load", cmdLoad, LW_TRUE, "[offset] <file>\n\tLoad file to memory at offset (or 0). Core must be in ICD."),
    MK_CMD("loadi", cmdLoadi, LW_TRUE, "[offset] <file>\n\tLoad file to IMEM at offset (or 0)."),
    MK_CMD("loadis", cmdLoadis, LW_TRUE, "[offset] <file>\n\tLoad file to IMEM at offset (or 0). Mark as *SECURE*"),
    MK_CMD("loadd", cmdLoadd, LW_TRUE, "[offset] <file>\n\tLoad file to DMEM at offset (or 0)."),
    MK_CMD("loade", cmdLoade, LW_TRUE, "[offset] <file>\n\tLoad file to EMEM at offset (or 0)."),
    MK_CMD("loadfb", cmdLoadfb, LW_TRUE, "[offset] <file>\n\tLoad file to framebuffer at offset (or 0). Core must be in ICD."),
    MK_CMD("dump", cmdDumpVa, LW_TRUE, "<address> <size>\n\tDump part of address space. Core must be in ICD."),
    MK_CMD("dumpi", cmdDumpi, LW_TRUE, "<offset> <size>\n\tDump IMEM."),
    MK_CMD("dumpd", cmdDumpd, LW_TRUE, "<offset> <size>\n\tDump DMEM."),
    MK_CMD("dumpe", cmdDumpe, LW_TRUE, "<offset> <size>\n\tDump EMEM."),
    MK_CMD("dumpfb", cmdDumpFb, LW_TRUE, "<offset> <size>\n\tDump part of framebuffer. Core must be in ICD."),
    MK_CMD("dumpcore", cmdDumpCore, LW_TRUE, "<address> [size] <file>\n\tWrite core dump to file. Size defaults to 4KiB. Core must be in ICD."),
#if LWWATCHCFG_IS_PLATFORM(WINDOWS) && !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
    MK_CMD("loadcore", cmdLoadCore, LW_TRUE, "[size] <file>\n\tLoad core dump and write it to file. Size of the buffer defaults to 8KiB. Works both on live crash and Windows minidump."),
#endif
    MK_CMD("dmesg", cmdDmesg, LW_TRUE, "[f] or <offset>\n\tDump (and empty if f) DMEM message ring buffer, or set DMEM message buffer offset."),
    MK_CMD("dmesge", cmdDmesge, LW_TRUE, "[f]\n\tDump (and empty if f) DMEM message ring buffer using ICD. You may need this on Bootrom+GA10x."),
    MK_CMD("rcsr", cmdReadCsr, LW_TRUE, "<number|name|name(index)>\n\tRead CSR <number>. Core must be in ICD."),
    MK_CMD("wcsr", cmdWriteCsr, LW_TRUE, "<number|name|name(index)>\n\tWrite CSR <number>. Core must be in ICD."),
    MK_CMD("state", cmdState, LW_TRUE, "\n\tDump RISC-V state."),
    MK_CMD("reload", cmdReload, LW_TRUE, "<file>\n\tResets RISC-V core, loads "
                                "new IMEM and keeps core in debug mode at pc=0."),
    MK_CMD("sec2", cmdSwitchSec2, LW_TRUE, "\n\tSwitch to SEC2 RISC-V instance."),
    MK_CMD("gsp", cmdSwitchGsp, LW_TRUE, "\n\tSwitch to GSP RISC-V instance."),
    MK_CMD("pmu", cmdSwitchPmu, LW_TRUE, "\n\tSwitch to PMU RISC-V instance. Supported only on GA100+."),
    MK_CMD("minion", cmdSwitchMinion, LW_TRUE, "\n\tSwitch to MINION RISC-V instance. Supported only on GH100+."),
    MK_CMD("tsec", cmdSwitchTsec, LW_TRUE, "\n\tSwitch to TSEC RISC-V instance. Supported only on T234+."),
    MK_CMD("lwdec0", cmdSwitchLwdec0, LW_TRUE, "\n\tSwitch to LWDEC0 RISC-V instance. Supported only on T234+ or GH100+."),
    MK_CMD("lwdec1", cmdSwitchLwdec1, LW_TRUE, "\n\tSwitch to LWDEC1 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec2", cmdSwitchLwdec2, LW_TRUE, "\n\tSwitch to LWDEC2 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec3", cmdSwitchLwdec3, LW_TRUE, "\n\tSwitch to LWDEC3 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec4", cmdSwitchLwdec4, LW_TRUE, "\n\tSwitch to LWDEC4 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec5", cmdSwitchLwdec5, LW_TRUE, "\n\tSwitch to LWDEC5 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec6", cmdSwitchLwdec6, LW_TRUE, "\n\tSwitch to LWDEC6 RISC-V instance. Supported only on GH100+."),
    MK_CMD("lwdec7", cmdSwitchLwdec7, LW_TRUE, "\n\tSwitch to LWDEC7 RISC-V instance. Supported only on GH100+."),
    MK_CMD("fsp", cmdSwitchFsp, LW_TRUE, "\n\tSwitch to FSP RISC-V instance. Supported only on GH100+."),
    MK_CMD("soe", cmdSwitchSoe, LW_TRUE, "\n\tSwitch to SOE RISC-V instance. Supported only on GH100+."),
    MK_CMD("mpu", cmdDumpMpu, LW_TRUE, "\n\tDump MPU mappings. In LS mode also dump WPR ID. Lwrrently broken on GA10x (HW)."),
    MK_CMD("pmp", cmdDumpPmp, LW_TRUE, "\n\tDump Core PMP mappings."),
    MK_CMD("iopmp", cmdDumpIoPmp, LW_TRUE, "<VA offset>\n\tDump IO-PMP mappings. Set VA offset to VA of priv base."),
    MK_CMD("bp", cmdDumpBreakpoint, LW_TRUE, "[command] {args}\n\tDump breakpoints. Commands: 's'et {idx} {addr} {flags}, 'c'lear {idx}, 'C'lear all"),
    MK_CMD("trace", cmdTrace, LW_FALSE, "[0|1|2|-|] \n\tTrace buffer handling. 0|1|2 modes,- disable, no arg dump."),

    MK_CMD("tasks", cmdTaskDbgList, LW_TRUE, "\n\tTask Debugger: List tasks."),
    MK_CMD("sessions", cmdTaskDbgSessions, LW_TRUE, "\n\tTask Debugger: List sessions."),

    MK_CMD("lsinfo", cmdShowCoreSelwrity, LW_TRUE, "\n\tShow core PRIV access level in LS mode."),

    MK_CMD("tdgdb", cmdTaskDbg, LW_FALSE, "<xTCB> \n\tTask Debugger: Start GDB stub without continue timeout. Attach immediately."),
    MK_CMD("tdbg", cmdTaskDbg2, LW_FALSE, "<xTCB> \n\tTask Debugger: Start GDB stub with 10s continue timeout. Attach immediately."),
    MK_CMD("tdwfh", cmdTaskDbg3, LW_FALSE, "<xTCB> \n\tTask Debugger: Start GDB stub with 10s continue timeout. Wait for halt."),

    MK_CMD("help", cmdHelp, LW_FALSE, "Help."),
    { NULL, NULL }
};

static LW_STATUS cmdHelp(const char *req)
{
    Command *cmd = commands;
    while (cmd->command)
    {
        if (!req || (strlen(req) == 0) || (strcmp(req, cmd->command) == 0))
            dprintf("\n%s %s\n", cmd->command, cmd->help);
        cmd++;
    }
    return LW_OK;
}

LW_STATUS riscvMain(const char *pArgs)
{
    const char *pCmd = NULL;
    int cmdLen = 0;
    Command *cmd = commands;
    LW_STATUS ret;

    if (!pRiscv[indexGpu].riscvIsSupported())
    {
        dprintf("RISC-V is not supported on this device.\n");
        return LW_ERR_NOT_SUPPORTED;
    }

    if (riscvInit() != LW_OK)
        return LW_ERR_GENERIC;

    // preconfigure for now
    config.bPrintGdbCommunication = 1;
    config.bPrintMemTransactions = 1;
    config.bPrintTargetCommunication = 0;

    dprintf("Build %s %s on %s RISC-V. Args: %s\n", __DATE__, __TIME__, pRiscvInstance->name, pArgs);

    pArgs = riscvGetToken(pArgs, &pCmd, &cmdLen);

    if (!pCmd || cmdLen == 0)
        return cmdHelp(NULL);

    while (cmd->command)
    {
        if (!strncmp(cmd->command, pCmd, cmdLen))
        {
            if (cmd->requiresLock) lwmutex_lock(icdLock);
            ret = cmd->handler(pArgs);
            if (cmd->requiresLock) lwmutex_unlock(icdLock);

            if (ret != LW_OK)
                dprintf("Command returned error: 0x%x\n", ret);
            if (ret == LW_ERR_ILWALID_OPERATION)
                return cmdHelp(pCmd);
            return ret;
        }
        cmd++;
    }

    dprintf("Command not found: %s\n", pCmd);
    return cmdHelp(NULL);
}

RISCV_CONFIG config = {0, };
