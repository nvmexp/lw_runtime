/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  seqdbg.c
 * @brief WinDbg Extension for the PMU Sequencer
 */

/* ------------------------ Includes ---------------------------------------- */

#include "os.h"
#include "hal.h"
#include "seq.h"
#include "pmu.h"
#include "parse.h"
#include "print.h"

/* ------------------------ Typedefs ---------------------------------------- */
/* ------------------------ Global Variables -------------------------------- */
/* ------------------------ Function Prototypes ----------------------------- */

static char *_seqGetNextWord (char **ppCmd);
static void  _seqPrintUsage  (void);
static void  _seqExecDump    (char *pCmd);

/* ------------------------ Public Functions -------------------------------- */

void 
seqExec
(
    char *pCmd
)
{
    char *pCmdName;
    pCmdName = _seqGetNextWord(&pCmd);
    if (strcmp(pCmdName, "dump") == 0)
    {
        _seqExecDump(pCmd);
    }
    else if (strcmp(pCmdName, "help") == 0)
    {
        _seqPrintUsage();
    }
    else
    {
        dprintf("lw: Unrecognized seq command: %s\n", pCmdName);
        dprintf("lw:\n");
        _seqPrintUsage();
    }
    return;
}

/* ------------------------ Private Functions ------------------------------- */

static void
_seqExecDump
(
    char *pCmd
)
{
    char      *pParams;
    char      *pFilename;
    FILE      *pFile;
    LwU8      *pBuffer;
    LwU64      address = 0;
    LwU64      size    = 0;
    LwU64      sizeRead;
    LwBool     bRaw    = LW_FALSE;
    LW_STATUS  status;

    pBuffer = (LwU8 *)malloc(4 * 1024);
    if (pBuffer == NULL)
    {
        dprintf("lw: Internal lwwatch error. Cannot allocate memory. "
                "Failing.");
        return;
    }

    pParams = (char *)pCmd;
    if (parseCmd(pCmd, "raw", 0, &pParams))
    {
        bRaw = LW_TRUE;
    }

    if (parseCmd(pCmd, "file", 1, &pParams))
    {
        pFilename = pParams;

        pFile = fopen(pFilename, "r");
        if (pFile == NULL)
        {
            dprintf("lw: Error: cannot open sequencer file: %s\n", pFilename);
            goto _seqExecDump_exit;
        }
        size = fread(pBuffer, 1, 4 * 1024, pFile);
        fclose(pFile);

    }
    else if (parseCmd(pCmd, "dmem", 2, &pParams))
    {
        GetExpressionEx(pParams, &address, &pParams);
        GetExpressionEx(pParams, &size   , &pParams);

        sizeRead = pPmu[indexGpu].pmuDmemRead(
                       (LwU32)address,    // addr
                       LW_TRUE,           // addr is VA
                       (LwU32)size / 4,   // length (dwords)
                       1,                 // port
                       (LwU32 *)pBuffer); // pDmem
        if (sizeRead != size / 4)
        {
            dprintf("lw: Error: unable to read %d bytes from DMEM at offset "
                    "0x%08x. Failing.\n", (LwU32)size, (LwU32)address);
            goto _seqExecDump_exit;
        }
    }
    else
    {
        if (parseCmd(pCmd, "rm", 2, &pParams))
        {
            GetExpressionEx(pParams, &address, &pParams);
            GetExpressionEx(pParams, &size   , &pParams);
        }
        else if (parseCmd(pCmd, "rm", 1, &pParams) ||
                (parseCmd(pCmd, "rm", 0, &pParams)))
        {
            dprintf("lw: Error. Must provide a RM (CPU) virtual-address "
                    "and a size for the script to read. Failing.\n");
            _seqPrintUsage();
            goto _seqExecDump_exit;
        }
        else if ((!GetExpressionEx(pParams, &address, &pParams)) ||
                 (!GetExpressionEx(pParams, &size   , &pParams)))
        {
            dprintf("lw: Error. Must provide a RM (CPU) virtual-address "
                    "and a size for the script to read. Failing.\n");
            _seqPrintUsage();
            goto _seqExecDump_exit;
        }

        status = readVirtMem(address, pBuffer, size, &sizeRead);
        if ((status != LW_OK) || (sizeRead != size))
        {
            dprintf("lw: Error: Unable to read %d bytes from memory. This may be "
                    "due to the elwiroment. Not all lwwatch elwironments are "
                    "capabile of reading RM virtual-memory. Please re-attempt "
                    "using a different dump mode (see -file and -dmem in usage "
                    "information).\n",
                    (LwU32)size);
            goto _seqExecDump_exit;
        }            
    }

    if (!bRaw)
    {
        seqDumpScript((LwU32 *)pBuffer, (LwU32)size);
    }
    else
    {
        printBuffer((char*)pBuffer, (LwU32)size, 0, 4);
    }

_seqExecDump_exit:
    free(pBuffer);
}

static char *
_seqGetNextWord
(
    char **ppCmd
)
{
    char *pCmd  = *ppCmd;
    char *pWord = NULL;

    // strip-off leading whitespace
    while (*pCmd == ' ')
    {
        pCmd++;
    }
    pWord = pCmd;

    // command-name ends at first whitespace character or EOS
    while ((*pCmd != ' ') && (*pCmd != '\0'))
    {
        pCmd++;
    }

    if (*pCmd != '\0')
    {
        *pCmd  = '\0';
        *ppCmd = pCmd + 1;
    }
    else
    {
        *ppCmd = pCmd;
    }
    return pWord;
}

static void
_seqPrintUsage(void)
{
    dprintf("lw: -------------------------------------------------------------------------------\n");
    dprintf("lw: Usage:\n");
    dprintf("lw:\n");
    dprintf("lw: !seq <function> [args]\n");
    dprintf("lw:\n");
    dprintf("lw:     Available functions:\n");
    dprintf("lw:        dump - Dump a sequencer script in human readable format\n");
    dprintf("lw:        help - Print this message\n");
    dprintf("lw:\n");
    dprintf("lw:     Function-Specific Usage:\n");
    dprintf("lw:        !seq dump <-rm|-file|-dmem> [options]\n");
    dprintf("lw:            -rm <address> <size>\n");
    dprintf("lw:                Source script from RM virtual-memory (default). This mode\n");
    dprintf("lw:                may not be supported in all lwwatch elwironments.\n");
    dprintf("lw:\n");
    dprintf("lw:            -file <filename>\n");
    dprintf("lw:                Source script from a sequencer dump file.\n");
    dprintf("lw:\n");
    dprintf("lw:            -dmem <address> <size>\n");
    dprintf("lw:                Source script from PMU data-memory (DMEM).\n");
    dprintf("lw:\n");
    dprintf("lw:            Options:\n");
    dprintf("lw:                -raw\n");
    dprintf("lw:                    Dump the raw (unparsed) script\n");
    dprintf("lw:\n");
    dprintf("lw: -------------------------------------------------------------------------------\n");
    dprintf("lw: Creating sequencer dump files in windbg:\n");
    dprintf("lw:\n");
    dprintf("lw:     Step1: Get the address of the RM's sequencer program buffer:\n");
    dprintf("lw:         kd> ?? pGpu->pSeq->pProgBuffer\n");
    dprintf("lw:         unsigned char * 0x87836008\n");
    dprintf("lw:\n");
    dprintf("lw:     Step2: Get the size of the buffer:\n");
    dprintf("lw:         kd> ?? pGpu->pSeq->pNextPtr - pGpu->pSeq->pProgBuffer\n");
    dprintf("lw:         int 0n612\n");
    dprintf("lw:\n");
    dprintf("lw:     Step3: Use the .writemem command to dump the buffer contents to a file\n");
    dprintf("lw:         kd> .writemem seq.bin 0x87836008 L0x264\n");
    dprintf("lw:         Writing 264 bytes.\n");
    dprintf("lw:\n");
    dprintf("lw: -------------------------------------------------------------------------------\n");
    dprintf("lw: Dumping sequencer scripts in RM virtual memory from windbg:\n");
    dprintf("lw:\n");
    dprintf("lw:     Step1: Get the address of the RM's sequencer program buffer:\n");
    dprintf("lw:         kd> ?? pGpu->pSeq->pProgBuffer\n");
    dprintf("lw:         unsigned char * 0x87836008\n");
    dprintf("lw:\n");
    dprintf("lw:     Step2: Get the size of the buffer:\n");
    dprintf("lw:         kd> ?? pGpu->pSeq->pNextPtr - pGpu->pSeq->pProgBuffer\n");
    dprintf("lw:         int 0n612\n");
    dprintf("lw:\n");
    dprintf("lw:     Step3: Dump the buffer:\n");
    dprintf("lw:         kd> !seq dump -rm 0x87836008 0x264\n");
    dprintf("lw:             - or - \n");
    dprintf("lw:         kd> !seq dump 0x87836008 0x264\n");
    dprintf("lw:\n");
}

