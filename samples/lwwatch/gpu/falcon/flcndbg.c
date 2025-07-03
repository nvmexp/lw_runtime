/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcndbg.c
 * @brief WinDbg Extension for Falcon.
 *
 * Related Files: flcnsym.c
 */

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "falcon.h"
#include "parse.h"
#include "print.h"
#include "lwsym.h"


/* ------------------------ Defines ---------------------------------------- */

static void _flcn_print_sdk_message(const char *func)
{
    dprintf("lw: %s - Please set LWW_MANUAL_SDK to a HW manual directory.\n", func);
    dprintf("lw: For example, " INC_DIR_EXAMPLE ".\n");
    dprintf("lw: \n");
    dprintf("lw: Be sure to use the same *software* branch from which the ucode was built. It\n");
    dprintf("lw: is from this path that the ucode binary path will be derived.\n");
}

#define FLCN_PRINT_SDK_MESSAGE() _flcn_print_sdk_message(__FUNCTION__)

#define FLCN_PRINT_NOTARGET_MESSAGE()                                         \
    dprintf("lw: The target Falcon engine is not set, please use \n"          \
            "lw: !flcn <engine name> to set the target engine.\n");

#define FLCN_MAX_LINE_LENGTH                     (256)

#define FLCN_FILE_SIZE_OVER_LINE_NUM_MIN         (20)

/*!
 * Define the number of digits of PC in hex
 */
#define FLCN_TRACEPC_PC_MAX_BITS                 (8)

/*
 * Get the LwU8 value for nth (from 0) hex bit (in char)
 */
#define FLCN_CHAR_TO_HEX(n, bit)                 (getLwU8FromChar(bit) * (1 << (4 * n)))

/* ------------------------ Types definitions ------------------------------ */

/* ------------------------ Globals ---------------------------------------- */
POBJFLCN thisFlcn = NULL;

/* ------------------------ Function Prototypes ---------------------------- */
static char*           flcnExecChompWordFromCmd   (char **ppCmd);
static void            flcnExecSympath            (char *pCmd);
static void            flcnExecLoad               (char *pCmd);
static void            flcnExelwnload             (void);
static LwBool          flcnAutosymDerive          (void);
static void            flcnExecExamineSymbol      (char  *pCmd);
static void            flcnExecDisplayMemory      (char  *pCmd, LwU8 width);
static void            flcnExecDmemWr             (char  *pCmd);
static void            flcnExecImport             (char  *pCmd);
static void            flcnExecRegisterDump       (char  *pCmd);
static void            flcnEmemRd                 (char  *pCmd);
static void            flcnEmemWr                 (char  *pCmd);
static void            flcnImemRd                 (char  *pCmd);
static void            flcnImemWr                 (char  *pCmd);
static void            flcnImblk                  (char  *pCmd);
static void            flcnImtag                  (char  *pCmd);
static void            flcnDmemRd                 (char  *pCmd);
static void            flcnDmBlk                  (char  *pCmd);
static void            flcnQueues                 (char  *pCmd);
static void            flcnSt                     (char  *pCmd);
static void            flcnTcb                    (char  *pCmd);
static void            flcnSched                  (char  *pCmd);
static void            flcnEvtq                   (char  *pCmd);
static void            flcnDmTag                  (char  *pCmd);
static void            flcnDmemOvl                (char  *pCmd);
static LwBool          flcnDmemRdPermitted        (void);
static LW_STATUS       flcntracepc                (char  *pCmd);
static LW_STATUS       flcnObjDumpLoad            (char *pCmd);

/* ------------------------ External Definitions --------------------------- */
extern BOOL GetSafeExpressionEx(char *Expression, LwU64 *Value, char **Remainder);
extern LwU64 GetSafeExpression(const char *lpExpression);

void
flcnExec
(
    char        *pCmd,
    POBJFLCN     pFlcn
)
{
    char        *pCmdName;

    pCmdName = flcnExecChompWordFromCmd(&pCmd);

    if (!pFlcn || !pFlcn->pFEIF || !pFlcn->pFCIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    thisFlcn = pFlcn;

    if (strcmp(pCmdName, "r") == 0)
    {
        flcnExecRegisterDump(pCmd);
    }
    else if (strcmp(pCmdName, "sympath") == 0)
    {
        flcnExecSympath(pCmd);
    }
    else if (strcmp(pCmdName, "load") == 0)
    {
        flcnExecLoad(pCmd);
    }
    else if (strcmp(pCmdName, "reload") == 0)
    {
        flcnExelwnload();
        flcnExecLoad(pCmd);
    }
    else if (strcmp(pCmdName, "unload") == 0)
    {
        flcnExelwnload();
    }
    else if (strcmp(pCmdName, "x") == 0)
    {
        flcnExecExamineSymbol(pCmd);
    }
    else if (strcmp(pCmdName, "dd") == 0)
    {
        flcnExecDisplayMemory(pCmd, 4);
    }
    else if (strcmp(pCmdName, "dw") == 0)
    {
        flcnExecDisplayMemory(pCmd, 2);
    }
    else if (strcmp(pCmdName, "db") == 0)
    {
        flcnExecDisplayMemory(pCmd, 1);
    }
    else if (strcmp(pCmdName, "import") == 0)
    {
        flcnExecImport(pCmd);
    }
    else if (strcmp(pCmdName, "imemrd") == 0)
    {
        flcnImemRd(pCmd);
    }
    else if (strcmp(pCmdName, "imemwr") == 0)
    {
        flcnImemWr(pCmd);
    }
    else if (strcmp(pCmdName, "imblk") == 0)
    {
        flcnImblk(pCmd);
    }
    else if (strcmp(pCmdName, "imtag") == 0)
    {
        flcnImtag(pCmd);
    }
    else if (strcmp(pCmdName, "queues") == 0)
    {
        flcnQueues(pCmd);
    }
    else if (strcmp(pCmdName, "dmemwr") == 0)
    {
        flcnExecDmemWr(pCmd);
    }
    else if (strcmp(pCmdName, "st") == 0)
    {
        flcnSt(pCmd);
    }
    else if (strcmp(pCmdName, "tcb") == 0)
    {
        flcnTcb(pCmd);
    }
    else if (strcmp(pCmdName, "sched") == 0)
    {
        flcnSched(pCmd);
    }
    else if (strcmp(pCmdName, "evtq") == 0)
    {
        flcnEvtq(pCmd);
    }
    else if (strcmp(pCmdName, "ememrd") == 0)
    {
        flcnEmemRd(pCmd);
    }
    else if (strcmp(pCmdName, "ememwr") == 0)
    {
        flcnEmemWr(pCmd);
    }
    else if (strcmp(pCmdName, "dmemrd") == 0)
    {
        flcnDmemRd(pCmd);
    }
    else if (strcmp(pCmdName, "dmblk") == 0)
    {
        flcnDmBlk(pCmd);
    }
    else if (strcmp(pCmdName, "dmtag") == 0)
    {
        flcnDmTag(pCmd);
    }
    else if (strcmp(pCmdName, "dmemovl") == 0)
    {
        flcnDmemOvl(pCmd);
    }
    else if (strcmp(pCmdName, "ldobjdump") == 0)
    {
        flcnObjDumpLoad(pCmd);
    }
    else if (strcmp(pCmdName, "tracepc") == 0)
    {
        if(flcntracepc(pCmd) != LW_OK)
        {
            dprintf("lw: TracePc Usage Help: <REQUIRED>  [OPTIONAL]\n");
            dprintf("lw: lw flcn \"ldobjdump <PATH_OF_.objdump>\" loads .objdumpfile.\n");
            dprintf("lw: lw flcn \"tracepc [PATH_OF_tracepc_file]\" exlwtes PC trace.\n");
            dprintf("lw: PCs to trace are loaded from PATH_OF_tracepc_file if given.\n");
            dprintf("lw:                         from flcn tracePC buffer otherwise.\n");
            dprintf("lw: Reload .objdump file after engine switching.\n");
        }
    }
    else
    {
        dprintf("lw: Unrecognized flcn command: %s\n", pCmdName);
        dprintf("lw:\n");
        FLCN_PRINT_USAGE_MESSAGE();
    }
}

static LW_STATUS
flcnParseObjdmp()
{
    LwU32        index;
    char         line[FLCN_MAX_LINE_LENGTH];
    LwU32        fileBufferLwrsor;
    LwU32        fnum = 0;
    LwU32        objdumpFileSize = thisFlcn->objdumpFileSize;
    LwU8*        pObjdumpBuffer = thisFlcn->pObjdumpBuffer;
    //First pass count functions
    fileBufferLwrsor = 0;
    index = 0;
    while (fileBufferLwrsor < objdumpFileSize)
    {
        if (index >= FLCN_MAX_LINE_LENGTH)
        {
            dprintf("lw: Error: Objdump File Line Too Long.\n");
            return LW_STATUS_LEVEL_ERR;
        }
        line[index] = pObjdumpBuffer[fileBufferLwrsor++];

        if (line[index] != '\n')
        {
            index++;
        }
        else
        {
            line[index + 1] = '\0';
            if ((index > 1) &&
                (((line[index - 1] == ':') && (line[index - 2] == '>')) ||
                 ((line[index - 2] == ':') && (line[index - 3] == '>'))))
            {
                thisFlcn->objdumpFileFuncN ++;
            }
            index  = 0;
        }
    }
    thisFlcn->ppObjdumpFileFunc = malloc(thisFlcn->objdumpFileFuncN * sizeof(LwU8 *));

    //Second pass collect functions
    fileBufferLwrsor  = 0;
    index = 0;
    while (fileBufferLwrsor < objdumpFileSize)
    {
        line[index] = pObjdumpBuffer[fileBufferLwrsor++];

        if (line[index] != '\n')
        {
            index++;
        }
        else
        {
            line[index + 1] = '\0';
            if ((index > 1) &&
                (((line[index - 1] == ':') && (line[index - 2] == '>')) ||
                 ((line[index - 2] == ':') && (line[index - 3] == '>'))))
            {
                LwU32 chrIdx = 0;
                while (isspace(line[chrIdx])) chrIdx ++;
                thisFlcn->ppObjdumpFileFunc[fnum] = malloc(strlen(line + chrIdx) + (size_t)1);
                memcpy(thisFlcn->ppObjdumpFileFunc[fnum], line + chrIdx,
                       strlen(line + chrIdx) + (size_t)1);
                ++ fnum;
            }
            index  = 0;
        }
    }
    return LW_OK;
}

static LW_STATUS
flcnObjDumpLoad
(
    char *pCmd
)
{
    LWSYM_STATUS status = LWSYM_OK;
    char *pFilename = flcnExecChompWordFromCmd(&pCmd);
    if ((pFilename == NULL) || (pFilename[0] == '\0'))
    {
        dprintf("lw: Error: Path for .objdump file needed. Terminating...\n");
        return LW_STATUS_LEVEL_ERR;
    }
    dprintf("lw: Loading objdumpfile: %s.\n", pFilename);
    if (thisFlcn->bObjdumpFileLoaded)
    {
        flcnTrpcClear(LW_TRUE, LW_FALSE);
    }
    status = lwsymFileLoad(pFilename, &(thisFlcn->pObjdumpBuffer),
                           &(thisFlcn->objdumpFileSize));

    if (status != LWSYM_OK)
    {
        dprintf("lw: Error status <%u>\n", (LwU32)status);
        dprintf("lw: Error reading file (%s)\n", lwsymGetStatusMessage(status));
        return LW_STATUS_LEVEL_ERR;
    }
    if(flcnParseObjdmp() != LW_OK)
    {
        flcnTrpcClear(LW_TRUE, LW_FALSE);
        dprintf("lw: Error: objdumpfile functions parse failed.\n");
    }
    else
    {
        thisFlcn->bObjdumpFileLoaded = LW_TRUE;
        dprintf("lw: objdumpfile functions loaded.\n");
    }
    return LW_OK;
}

static LW_STATUS
flcnParseTracepc
(
    const LwU8 *pFileBuffer,
    LwU32 pFilesize
)
{
    LwU32        fileBufferLwrsor;
    LwU32        pcnum = 0;

    //Clear the extTracepc
    flcnTrpcClear(LW_FALSE, LW_TRUE);

    //First pass count PCs
    fileBufferLwrsor = 0;
    while (fileBufferLwrsor < pFilesize)
    {
        if (pFileBuffer[fileBufferLwrsor] == '0' &&
            pFileBuffer[fileBufferLwrsor + 1] == 'x')
        {
            thisFlcn->extTracepcNum ++;
            fileBufferLwrsor += (FLCN_TRACEPC_PC_MAX_BITS + 2);
        }
        else
        {
            fileBufferLwrsor ++;
        }
    }

    thisFlcn->pExtTracepcBuffer = malloc(thisFlcn->extTracepcNum * sizeof(LwU8 *));

    //Second pass collect PCs
    fileBufferLwrsor = 0;
    while (fileBufferLwrsor < pFilesize)
    {
        if (pFileBuffer[fileBufferLwrsor] == '0' &&
            pFileBuffer[fileBufferLwrsor + 1] == 'x')
        {
            thisFlcn->pExtTracepcBuffer[pcnum] = malloc((size_t)FLCN_TRACEPC_PC_MAX_BITS + (size_t)1);
            memcpy(thisFlcn->pExtTracepcBuffer[pcnum], pFileBuffer + fileBufferLwrsor + 2,
                   (size_t)FLCN_TRACEPC_PC_MAX_BITS);
            thisFlcn->pExtTracepcBuffer[pcnum][FLCN_TRACEPC_PC_MAX_BITS] = '\0';
            ++ pcnum;
            fileBufferLwrsor += (FLCN_TRACEPC_PC_MAX_BITS + 2);
        }
        else
        {
            fileBufferLwrsor ++;
        }
    }
    return LW_OK;
}

static LwU8
getLwU8FromChar
(
    char bitChr
)
{
    if ((bitChr >= '0') && (bitChr <= '9'))
    {
        return bitChr - '0';
    }
    else
    {
        switch (bitChr)
        {
            case 'a': return 10;
            case 'b': return 11;
            case 'c': return 12;
            case 'd': return 13;
            case 'e': return 14;
            case 'f': return 15;
            default :
                {
                    dprintf("lw: Error: PC contains unexpected digit.\n");
                    dprintf("lw:        Check your source file.\n");
                    return 16;
                }
        }
    }
}

static LwU32
flcntracepcFindPc
(
    LwU32 PC
)
{
    extern  POBJFLCN thisFlcn;
    LwU8    pcstr[(size_t)FLCN_TRACEPC_PC_MAX_BITS + (size_t)1];
    LwU32   ln = 0;
    LwU32   rn = thisFlcn->objdumpFileFuncN - 1;
    LwU32   fileFuncNum = rn + 1;
    LwU32   midn;
    LwU32   index;
    LwS32   cmpResult;

    while (1)
    {
        LwU32   tempPc = 0;
        midn = (ln + rn) / 2;
        memcpy(pcstr, thisFlcn->ppObjdumpFileFunc[midn], FLCN_TRACEPC_PC_MAX_BITS);
        pcstr[FLCN_TRACEPC_PC_MAX_BITS] = '\0';
        for (index = 0; index < FLCN_TRACEPC_PC_MAX_BITS; ++ index)
        {
            tempPc += FLCN_CHAR_TO_HEX(index, pcstr[FLCN_TRACEPC_PC_MAX_BITS - 1 - index]);
        }
        cmpResult = PC - tempPc;
        if (cmpResult >= 0)
        {
            if (ln == rn) {
                return rn;
            }
            ln = midn;
        }
        else
        {
            if (ln == rn) {
                return rn - 1;
            }
            rn = midn;
        }
        if ((rn - ln) == 1)
        {
            if (rn < fileFuncNum - 1)
            {
                return ln;
            }
            else
            {
                ln = rn;
            }
        }
    }
}

static LW_STATUS
flcntracepc
(
    char  *pCmd
)
{
    LwU32           index = 0;
    LwU32           lwrrentPC;
    LwU32           i;
    LwU32           funcID;
    LwU32           maxIdx = 0;
    LwU32           pcCnt;
    LwU32           pcFileSize;
    LwBool          bCompressed = LW_FALSE;
    LwBool          bExtPC = LW_FALSE;
    LwBool          bPc2Func;
    LwU8*           pTrpcFB;
    char*           pExtPath = NULL;
    char**          ppObjdumpFileFunc = NULL;
    char*           pLine = NULL;
    LWSYM_STATUS    status;

    if (thisFlcn->bObjdumpFileLoaded)
    {
        bPc2Func = LW_TRUE;
    }
    else
    {
        bPc2Func = LW_FALSE;
        dprintf("lw: objdumpFile not loaded. PCs will not be mapped to functions.\n");
    }
    ppObjdumpFileFunc = thisFlcn->ppObjdumpFileFunc;
    pExtPath = flcnExecChompWordFromCmd(&pCmd);
    bExtPC = (pExtPath != NULL) && (pExtPath[0] != '\0');

    if (!bExtPC)
    {
        maxIdx = pFalcon[indexGpu].falconTrpcGetMaxIdx(thisFlcn->engineBase);
        bCompressed = pFalcon[indexGpu].falconTrpcIsCompressed(thisFlcn->engineBase);
    }
    else
    {
        status = lwsymFileLoad(pExtPath, &pTrpcFB, &pcFileSize);
        if (status != LWSYM_OK)
        {
            dprintf("lw: Error status is %u\n", (LwU32)status);
            dprintf("lw: Error reading file (%s)\n", lwsymGetStatusMessage(status));
            return LW_STATUS_LEVEL_ERR;
        }
        else
        {
            if (flcnParseTracepc(pTrpcFB, pcFileSize) != LW_OK)
            {
                dprintf("lw: Error: Parse Trace PC file failed. Terminating...\n");
                return LW_STATUS_LEVEL_ERR;
            }
            free(pTrpcFB);
            pTrpcFB = NULL;
            maxIdx = thisFlcn->extTracepcNum - 1;
        }
    }
    dprintf("lw: TRACE PC FORMAT:%s. \n", bCompressed ? "COMPRESSED" : "UNCOMPRESSED");
    dprintf("lw: TRACE PC SOURCE:%s. \n", bExtPC ? "USER" : "FLCN");
    if (!bCompressed)
    {
        dprintf("lw:        TraceBufferIdx       PC                 In<Func>.\n");
    }
    else
    {
        dprintf("lw:        TraceBufferIdx       PC                 Count            In<Func>.\n");
    }
    for (i = 0; i <= maxIdx; ++ i)
    {
        pcCnt = 0;
        lwrrentPC = 0;
        //Get PC
        if (!bExtPC)
        {
            lwrrentPC = pFalcon[indexGpu].falconTrpcGetPC(thisFlcn->engineBase, i,
                                                          bCompressed ? &pcCnt : NULL);
        }
        else
        {
            char    PCstr[FLCN_TRACEPC_PC_MAX_BITS + 1];
            memcpy(PCstr, thisFlcn->pExtTracepcBuffer[i],
                   (size_t)FLCN_TRACEPC_PC_MAX_BITS);
            PCstr[FLCN_TRACEPC_PC_MAX_BITS] = '\0';
            for (index = 0; index < FLCN_TRACEPC_PC_MAX_BITS; ++ index)
            {
                lwrrentPC += FLCN_CHAR_TO_HEX(index, PCstr[FLCN_TRACEPC_PC_MAX_BITS - 1 - index]);
            }
        }
        //Map to function if applicable
        if (bPc2Func)
        {
            funcID = flcntracepcFindPc(lwrrentPC);
            pLine = malloc(strlen(ppObjdumpFileFunc[funcID]) - (size_t)1);
            memcpy(pLine, ppObjdumpFileFunc[funcID],
                   strlen(ppObjdumpFileFunc[funcID]) - (size_t)2);
            pLine[strlen(ppObjdumpFileFunc[funcID]) - 2] = '\0';
            index = 0;
            while (!isspace(pLine[index ++])) {}
        }
        if (!bCompressed){
            dprintf("lw:        0x%02x                 0x%08x         %s.\n",
                     i, lwrrentPC, bPc2Func ? (pLine + index) : "N/A");
        }
        else
        {
            dprintf("lw:        0x%02x                 0x%08x         %u                    %s.\n",
                     i, lwrrentPC, pcCnt, bPc2Func ? (pLine + index) : "N/A");
        }
        if (bPc2Func)
        {
            free(pLine);
        }
    }
    dprintf("lw: Total PC traced: %u \n", i);
    return LW_OK;
}

static void
flcnSched
(
    char *pCmd
)
{
    BOOL bTable = TRUE;

    // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    if (parseCmd(pCmd, "l", 0, NULL))
    {
        bTable = FALSE;
    }

    flcnRtosSchedDump(bTable);
}


static char *
flcnExecChompWordFromCmd
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

    // command-name ends at first non-whitespace character
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
flcnExecSympath
(
    char   *pCmd
)
{
    char   *pToken;

    pToken = strtok(pCmd, " ");
    if (pToken != NULL)
    {
        memset(thisFlcn->symPath, '\0', sizeof(thisFlcn->symPath));
        strcpy(thisFlcn->symPath, pToken);
    }
    if (*thisFlcn->symPath != '\0')
    {
        dprintf("lw: Symbol search path is: %s\n", thisFlcn->symPath);
        thisFlcn->bSympathSet = TRUE;
    }
    else if (flcnAutosymDerive())
    {
        thisFlcn->bSympathSet = TRUE;
    }
    return;
}

static LwBool
flcnAutosymDerive(void)
{
    const char *pManualDir = NULL;
    const char *pSymFilePath = NULL;

    if (thisFlcn->bSympathAutoDerived)
    {
        return TRUE;
    }

    if ((pManualDir = getelw("LWW_MANUAL_SDK")) == NULL)
    {
        FLCN_PRINT_SDK_MESSAGE();
        return FALSE;
    }

    pSymFilePath = thisFlcn->pFEIF->flcnEngGetSymFilePath();

    memset(thisFlcn->symPath, '\0', sizeof(thisFlcn->symPath));

    strcpy(thisFlcn->symPath, pManualDir);
    strcat(thisFlcn->symPath, pSymFilePath);

    thisFlcn->bSympathAutoDerived = TRUE;
    thisFlcn->bSympathSet = TRUE;
    dprintf("lw: Symbol search path is: %s\n", thisFlcn->symPath);
    dprintf("lw:    (derived from LWW_MANUAL_SDK)\n");
    return TRUE;
}

static void
flcnExecLoad
(
    char *pCmd
)
{
    LwU32        ucodeVersion;
    size_t       pathLength;
    const char  *pUcodeName;
    char        *pFilename;
    char        *pTemp;

    if (!thisFlcn->pFEIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    //
    // If a symbol-path is not set, try to derive one from elw-vars. If that
    // fails, just abort.
    //
    if ((!thisFlcn->bSympathSet) && (!flcnAutosymDerive()))
    {
        return;
    }

    // unload any information that's lwrrently loaded
    if (thisFlcn->bSymLoaded)
    {
        flcnExelwnload();
    }

    ucodeVersion = thisFlcn->pFCIF->flcnUcodeGetVersion(thisFlcn->engineBase);
    // determine the version of the ucode/application lwrrently running
    if ((pCmd != NULL) && (pCmd[0] != '\0'))
    {
        flcnSymLoad(pCmd, ucodeVersion);
    }
    else
    {
        pUcodeName   = thisFlcn->pFEIF->flcnEngUcodeName();

        pathLength  = strlen(thisFlcn->symPath);
        pathLength += strlen(pUcodeName);
        pathLength += 3 + 1; // 3 bytes for .nm and 1 byte for null

        pFilename = (char *)malloc(pathLength);
        if (pFilename == NULL)
            return;
        strcpy(pFilename, thisFlcn->symPath);
        strcat(pFilename, pUcodeName);
        pTemp = pFilename + strlen(pFilename);

        // load the FALCON symbols from the nm-file.
        strcpy(pTemp, ".nm");
        flcnSymLoad(pFilename, ucodeVersion);

        free(pFilename);
    }
}

static void
flcnExelwnload(void)
{
    flcnSymUnload();
    return;
}

static void
flcnExecExamineSymbol
(
    char *pCmd
)
{
    if (!thisFlcn->bSymLoaded)
    {
        dprintf("lw: Error: %s engine symbol information is not loaded.  Use:\n",thisFlcn->engineName);
        dprintf("lw: !flcn load\n");
        return;
    }

    flcnSymDump(pCmd, TRUE);
}

// flcn dd [options] Range
static void
flcnExecDisplayMemory
(
    char  *pCmd,
    LwU8   width
)
{
    LwU64 offset;
    LwU64 lengthInBytes = 0x10;
    LwU64 port = thisFlcn->pFEIF->flcnEngGetDmemAccessPort();

    if (!thisFlcn->pFEIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    if (!GetExpressionEx(pCmd, &offset, &pCmd))
    {
        return;
    }

    while (*pCmd != '\0')
    {
        if (*pCmd == 'L')
        {
            pCmd++;
            if (!GetExpressionEx(pCmd, &lengthInBytes, &pCmd))
                return;
        }
        pCmd++;
    }

    lengthInBytes *= width;
    flcnDmemDump(thisFlcn->pFEIF, (LwU32)offset, (LwU32)lengthInBytes, (LwU8)port, width);

}

static void
flcnDmemRd
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32    engineBase      = 0x0;
    LwU64    offset          = 0x0;
    LwU64    lengthInBytes   = 0x80;
    LwU64    port            = 0x0;
    LwU32    memSize         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU32    length          = 0x0;
    LwU32    dmemVaBound     = 0x0;
    LwBool   bIsAddrVa       = LW_FALSE;
    LwU32   *buffer          = NULL;

    // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    if (pCmd[0] == '\0')
    {
        dprintf("lw: Usage: !flcn dmemrd <offset> [length(bytes)] [port]\n");
        dprintf("lw: No args specified, defaulted to offset"
                " 0x%04x and length 0x%04x bytes.\n",
                (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetExpressionEx(pCmd, &offset, &pCmd))
    {
        if (GetExpressionEx(pCmd, &lengthInBytes, &pCmd))
        {
            GetExpressionEx(pCmd, &port, &pCmd);
        }
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnDmemGetSize(engineBase);
        numPorts   = pFCIF->flcnDmemGetNumPorts(engineBase);
        port       = pFEIF->flcnEngGetDmemAccessPort();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        dmemVaBound = pFCIF->flcnDmemVaBoundaryGet(engineBase);
        if ((LW_FLCN_DMEM_VA_BOUND_NONE != dmemVaBound) &&
            (LwU32)offset >= dmemVaBound)
        {
            if ((offset < dmemVaBound) &&
                ((offset + length * 4) > dmemVaBound))
            {
                dprintf("lw:\tError: Attempt to read across DMEM VA boundary.\n");
                return;
            }
            bIsAddrVa = LW_TRUE;
        }
        else
        {
            // Prevent allocating too much unused memory in temp buffer
            if ((LwU32)offset >= memSize)
            {
                dprintf("lw: %s: offset 0x%04x is too large (DMEM size 0x%04x)\n",
                        __FUNCTION__, (LwU32)offset, (LwU32)memSize);
                return;
            }

            // Prevent allocating too much unused memory in temp buffer
            if ((LwU32)(offset + lengthInBytes) >= memSize)
            {
                dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
                lengthInBytes = memSize - offset;
            }
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the DMEM
        length = pFCIF->flcnDmemRead(engineBase,
                                     (LwU32)offset,
                                     bIsAddrVa,
                                     (LwU32)lengthInBytes / sizeof(LwU32),
                                     (LwU32)port, buffer);

        // Dump out the DMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping %s DMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    thisFlcn->engineName,
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }

        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

static void
flcnDmBlk
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32       engineBase          = 0x0;
    BOOL        bListAll            = TRUE;
    LwU64       userIndex           = ~0x0;
    LwU32       blockIndex          = 0x0;
    LwU32       numBlocks           = 0x0;
    LwU32       tagWidth            = 0x0;
    FLCN_BLOCK  blockInfo;

   // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    // Get the block index
    GetExpressionEx(pCmd, &userIndex, &pCmd);
    bListAll    = (userIndex == (LwU64)(~0));
    blockIndex  = (LwU32)userIndex;

    MGPU_LOOP_START;
    {
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        numBlocks  = pFCIF->flcnDmemGetSize(engineBase);
        tagWidth   = pFCIF->flcnDmemGetTagWidth(engineBase);


        if (!bListAll)
        {
            if (blockIndex >= numBlocks)
            {
                dprintf("lw: Block index 0x%x is invalid (max 0x%x).\n",
                        blockIndex, numBlocks - 1);
                return;
            }
            else
            {
                numBlocks = blockIndex + 1;
            }
        }
        else
        {
            blockIndex = 0;
        }

        dprintf("lw:\tDumping DMEM code block status\n");
        dprintf("lw:\t--------------------------------------------------\n");

        // Loop through all the blocks dumping out information
        for (; blockIndex < numBlocks; blockIndex++)
        {
            if (pFCIF->flcnDmemBlk(engineBase, blockIndex, &blockInfo))
            {
                dprintf("lw:\tBlock 0x%02x: tag=0x%02x, valid=%d, pending=%d, "
                        "secure=%d\n",
                        blockIndex, blockInfo.tag, blockInfo.bValid,
                        blockInfo.bPending, blockInfo.bSelwre);
            }

        }

        dprintf("\n");
    }
    MGPU_LOOP_END;
}

static void
flcnDmTag
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32       engineBase          = 0x0;
    LwU64       addr                = 0x0;
    LwU32       tagWidth            = 0x0;
    LwU32       maxAddr             = 0x0;
    FLCN_TAG    tagInfo;

    if (pCmd[0] == '\0')
    {
        dprintf("lw:\tUsage: !lw.flcn -<engine> dmtag <code addr>\n");
        return;
    }

   // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    // Get the block address to look up
    GetExpressionEx(pCmd, &addr, &pCmd);

    MGPU_LOOP_START;
    {
        //
        // Get the tag width and notify user when code address entered
        // is larger than the maximum allowed by tag width.
        // Note that bits 0-7 are offsets into the block and 8-7+TagWidth
        // define the tag bits.
        //
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        tagWidth   = pFCIF->flcnDmemGetTagWidth(engineBase);

        maxAddr = BIT(tagWidth+8) - 1;

        if (addr > maxAddr)
        {
            dprintf("lw: %s: Address 0x%04x is too large (max 0x%04x)\n",
                    __FUNCTION__, (LwU32)addr, maxAddr);
            return;
        }

        dprintf("lw: Dumping block info for address 0x%x, Tag=0x%x\n",
                (LwU32)addr, ((LwU32)addr) >> 8);

        if (pFCIF->flcnDmemTag(engineBase, (LwU32)addr, &tagInfo))
        {
            switch (tagInfo.mapType)
            {
            case FALCON_TAG_UNMAPPED:
                dprintf("lw:\tTag 0x%02x: Not mapped to a block", (LwU32)addr >> 8);
                break;
            case FALCON_TAG_MULTI_MAPPED:
            case FALCON_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%d",
                        tagInfo.blockInfo.tag, tagInfo.blockInfo.blockIndex,
                        tagInfo.blockInfo.bValid, tagInfo.blockInfo.bPending,
                        tagInfo.blockInfo.bSelwre);
                break;
            }
            if (tagInfo.mapType == FALCON_TAG_MULTI_MAPPED)
            {
                dprintf(" (multiple)");
            }
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;
}

static void
flcnDmemOvl
(
    char *pCmd
)
{
    // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        dprintf("DMEM Read is not permitted.  Aborting!\n");
        return;
    }

    MGPU_LOOP_START;
    {
        if (thisFlcn && thisFlcn->pFCIF)
        {
            if (!thisFlcn->bSymLoaded)
            {
                dprintf("lw: symbols not loaded, loading automatically...\n");
                flcnExec("load", thisFlcn);
            }
            dprintf("lw: dumping dmem overlays\n");
            flcnRtosDmemOvlDumpAll();
        }
    }
    MGPU_LOOP_END;
}

// flcn dmemwr <offset> <value> <-w width(1=byte, 2=half-word, 4=word)> <-l length (number of entries of width w>
static void
flcnExecDmemWr
(
    char  *pCmd
)
{
    LwU64 offset;
    LwU64 value;
    LwU64 width  = sizeof(LwU32); // default = 4 (word)
    LwU64 length = 0x1;           // default = 1 (1 word)
    LwU64 port   = thisFlcn->pFEIF->flcnEngGetDmemAccessPort();
    char  *pParams = (char *)pCmd;

    if (!thisFlcn->pFEIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    if (pCmd[0] == '\0')
    {
        dprintf("lw: Incorrect usage\n");
        dprintf("lw: usage: !flcn <-supported falcon name> dmemwr <offset> <value> <-w width> <-l length>\n");
        dprintf("lw: usage: width (1=byte, 2=half word, 4=word) and length (number of entries of width w) "
                    "are optional. default width = 0x4 (word), length = 0x1 (one word)\n");
        return;
    }

    // Extract optional arguments first
    if (parseCmd(pCmd, "w", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &width, &pParams);
        if (width != sizeof(LwU32) && width != sizeof(LwU16) && width != sizeof(LwU8))
        {
            // Incorrect usage
            dprintf("lw: width not supported. Must be 1 (byte), 2 (half-word), or 4 (word)\n");
            return;
        }
    }

    if (parseCmd(pCmd, "l", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &length, &pParams);
    }

    // Read in the <offset> <value> in that order.
    if (GetExpressionEx(pCmd, &offset, &pCmd))
    {
        GetExpressionEx(pCmd, &value, &pCmd);
    }
    else
    {
        // Incorrect usage
        dprintf("lw: Incorrect usage\n");
        dprintf("lw: usage: !flcn <-supported falcon name> dmemwr <offset> <value> <-w width> <-l length>\n");
        dprintf("lw: usage: width (1=byte, 2=half word, 4=word) and length (number of entries of width w) "
                    "are optional. default width = 0x4 (word), length = 0x1 (one word)\n");
        return;
    }
    flcnDmemWrWrapper(thisFlcn->pFEIF, (LwU32)offset, (LwU32)value, (LwU32)width, (LwU32)length, (LwU8)port);
}

static void
flcnExecRegisterDump
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES *pFCIF = thisFlcn->pFCIF;
    LwU32        val32;
    LwU32        count;
    LwU32        i;

    if (!thisFlcn->pFEIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    dprintf("lw: Dumping Falcon Internal Registers...\n");
    dprintf("lw:\n");

    //
    // Dump the PC first including the nearest code symbol name (if symbols are
    // loaded).
    //
    val32 = pFCIF->flcnGetRegister(thisFlcn->engineBase, LW_FLCN_REG_PC);
    dprintf("lw:   pc=0x%08x", val32);

    if ((val32 != 0xFFFFFFFF) && flcnSymCheckIfLoaded())
    {
        PFLCN_SYM pMatches;
        PFLCN_SYM pSym = NULL;

        pMatches = flcnSymResolve(val32);
        while (pMatches != NULL)
        {
            if ((pMatches->section == 'T') || (pMatches->section == 't'))
            {
                pSym = pMatches;
                break;
            }
            else if (pMatches->section == 'A')
            {
                pSym = pMatches;
            }
            pMatches = pMatches->pTemp;
        }

        if (pSym != NULL)
        {
            dprintf(" <%s+0x%x>", pSym->name, val32 - pSym->addr);
        }
    }

    dprintf("\n");

    // dump SP
    val32 = pFCIF->flcnGetRegister(thisFlcn->engineBase, LW_FLCN_REG_SP);
    dprintf("lw:   sp=0x%08x\n", val32);
    dprintf("lw:\n");
    dprintf("lw: ");

    // dump other registers
    count = sizeof(FLCN_REG_TABLE) / sizeof(FLCN_REG_INFO);
    for (i = 0; i < count; i++)
    {
        val32 = pFCIF->flcnGetRegister(thisFlcn->engineBase, FLCN_REG_TABLE[i].regIdx);
        dprintf("%4s=0x%08x", FLCN_REG_TABLE[i].pName, val32);
        if (((i + 1) % 4 == 0) && (i != 0))
        {
            dprintf("\nlw: ");
        }
        else if ((i + 1) != count)
        {
            dprintf(", ");
        }
    }
    dprintf("\nlw:\n");

}

static void
flcnExecImport
(
    char *pCmd
)
{
    FILE  *pFile;
    char  *pFilename;
    LwS32  b;
    LwU32  w = 0;
    LwU32  n = 0;
    LwU32  c = 0;
    LwU64  offset;
    LwU32  start;
    LwBool bWriteSuccess = LW_TRUE;

    if (!GetExpressionEx(pCmd, &offset, &pCmd))
    {
        dprintf("lw: offset not given\n");
        dprintf("lw: usage: !flcn import <offset> <filename>\n");
        return;
    }
    start = (LwU32)offset;

    pFilename = strtok(pCmd, " ");
    if (pFilename == NULL)
    {
        dprintf("lw: import filename not given\n");
        dprintf("lw: usage: !flcn import <offset> <filename>\n");
        return;
    }

    pFile = fopen(pFilename, "rb");
    if (pFile == NULL)
    {
        dprintf("lw: Error: cannot open import file: %s\n", pFilename);
        return;
    }

    dprintf("lw:\n");
    dprintf("lw: Import data file: %s\n", pFilename);
    dprintf("lw: ");

    do
    {
        dprintf(".");
        b  = fgetc(pFile);
        w |= (b << (n * 8));
        n  = (n + 1) % 4;
        if (n == 0)
        {
            if (thisFlcn->pFCIF->flcnDmemWrite(thisFlcn->engineBase,
                                               (LwU32)offset,
                                               LW_FALSE,
                                               w, 0x4, 0x1, 0x1) != 4)
            {
                bWriteSuccess = LW_FALSE;
                break;
            }
            offset += 4;
            w = 0;
        }
        c++;
    }
    while (b != EOF);
    fclose(pFile);

    c--;
    dprintf("\n");
    dprintf("lw:\n");
    if (bWriteSuccess)
    {
        dprintf("lw: OK\n");
        dprintf("lw: Imported %d bytes to 0x%04x-0x%04x %s DMEM\n",
            c, start, start + c - 1, thisFlcn->engineName);
    }
    else
    {
        dprintf("lw: FAILED to write at offset 0x%x\n", (LwU32)offset);
    }
    dprintf("lw:\n");
}

// flcn ememrd <offset> [length(bytes)] [port]
static void
flcnEmemRd
(
    char *pCmd
)
{
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32    memSize         = 0x0;
    LwU32    startEmem       = 0x0;
    LwU32    endEmem         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU64    offset          = 0x0;
    LwU64    lengthInBytes   = 0x80;
    LwU64    port            = 0x0;
    LwU32    length          = 0x0;
    LwU32   *pBuf            = NULL;

    MGPU_LOOP_START;
    {
        // Check that EMEM is supported
        pFEIF = thisFlcn->pFEIF;
        if (pFEIF->flcnEngEmemGetSize() == 0)
        {
            dprintf("lw: EMEM not supported by engine\n");
            return;
        }

        // Set engine-specific defaults
        numPorts  = pFEIF->flcnEngEmemGetNumPorts();
        memSize   = pFEIF->flcnEngEmemGetSize();
        startEmem = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
        endEmem   = startEmem + memSize;
        offset    = startEmem;

        if (pCmd[0] == '\0')
        {
            dprintf("lw: Usage: !flcn ememrd <offset> [len(bytes)] [port]\n");
            dprintf("lw: No args specified, defaulted to offset 0x%x"
                    " and length 0x%04x bytes.\n",
                    (LwU32)offset, (LwU32)lengthInBytes);
            dprintf("lw: EMEM located in address range [0x%08x,0x%08x)\n",
                    startEmem, endEmem);
        }

        // Read in the <offset> [length] [port], in that order, if present
        if (GetExpressionEx(pCmd, &offset, &pCmd))
        {
            if (GetExpressionEx(pCmd, &lengthInBytes, &pCmd))
            {
                GetExpressionEx(pCmd, &port, &pCmd);
            }
        }

        // Tidy up the length and offset to be 4-byte aligned
        lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
        offset          = offset & ~3ULL;

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent reading outside of EMEM
        if ((LwU32)offset < startEmem || (LwU32)offset >= endEmem)
        {
            dprintf("lw: %s: Offset 0x%0x not in EMEM aperature [0x%x,0x%x)\n",
                    __FUNCTION__, (LwU32)offset, startEmem, endEmem);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) > endEmem)
        {
            dprintf("lw: %s: length larger then memory size,"
                    " truncating to fit\n", __FUNCTION__);
            lengthInBytes = endEmem - offset;
        }

        // Create a temporary buffer to store data
        pBuf = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (pBuf == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n",
                    __FUNCTION__);
            return;
        }

        // Actually read the EMEM
        length = pFEIF->flcnEngEmemRead((LwU32)offset,
                                        (LwU32)lengthInBytes / sizeof(LwU32),
                                        (LwU32)port,
                                        pBuf);

        // Dump out the EMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping %s EMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    thisFlcn->engineName,
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)pBuf, length * sizeof(LwU32), offset, 0x4);
        }

        // Cleanup after ourselves
        free((void*)pBuf);
    }
    MGPU_LOOP_END;
}

// flcn ememwr <offset> <value> [-w <width>(bytes)] [-l <length>(units of width)] [port]
static void
flcnEmemWr
(
    char *pCmd
)
{
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32 memSize       = 0x0;
    LwU32 startEmem     = 0x0;
    LwU32 endEmem       = 0x0;
    LwU64 offset        = 0x0;
    LwU64 length        = 0x1;
    LwU64 width         = 0x4;
    LwU64 value         = 0x0;
    LwU64 port          = 0x0;
    LwU32 numPorts      = 0x0;
    LwU32 bytesWritten  = 0x0;
    char  *pParams;

    MGPU_LOOP_START;
    {
        // Check that EMEM is supported
        pFEIF = thisFlcn->pFEIF;
        if (pFEIF->flcnEngEmemGetSize() == 0)
        {
            dprintf("lw: EMEM not supported by engine\n");
            return;
        }

        memSize   = pFEIF->flcnEngEmemGetSize();
        startEmem = pFEIF->flcnEngEmemGetOffsetInDmemVaSpace();
        endEmem   = startEmem + memSize;

        if (pCmd[0] == '\0')
        {
            dprintf("lw: Offset not specified.\n");
            dprintf("lw: Usage: !flcn ememwr <offset> <value>"
                    "[-w <width>(bytes)] "
                    "[-l <length>(units of width)] [-p <port>]\n");
            dprintf("lw: EMEM located in address range [0x%08x,0x%08x)\n",
                    startEmem, endEmem);
            return;
        }

        // extract optional arguments first
        if (parseCmd(pCmd, "w", 1, &pParams))
        {
            GetExpressionEx(pParams, &width, &pParams);
        }

        if (parseCmd(pCmd, "l", 1, &pParams))
        {
            GetExpressionEx(pParams, &length, &pParams);
        }

        if (parseCmd(pCmd, "p", 1, &pParams))
        {
            GetExpressionEx(pParams, &port, &pParams);
        }

        // Read in the <offset> <value>, in that order, if present
        if (!GetExpressionEx(pCmd, &offset, &pCmd))
        {
            dprintf("lw: Value not specified.\n");
            dprintf("lw: Usage: !flcn ememwr <offset> <value> [-w <width>(bytes)] "
                    "[-l <length>(units of width)] [-p <port>]\n");
            return;
        }
        GetExpressionEx(pCmd, &value, &pCmd);

        // Get the size of the EMEM and number of EMEM ports
        numPorts   = pFEIF->flcnEngEmemGetNumPorts();

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the EMEM
        dprintf("lw:\tWriting EMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at EMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFEIF->flcnEngEmemWrite(
                           (LwU32)offset,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);

        dprintf("lw: number of bytes written: 0x%x\n", bytesWritten);
    }
    MGPU_LOOP_END;
}

static void
flcnImemRd
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32    engineBase      = 0x0;
    LwU64    offset          = 0x0;
    LwU64    lengthInBytes   = 0x80;
    LwU64    port            = 0x0;
    LwU32    memSize         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU32    length          = 0x0;
    LwU32   *buffer          = NULL;

    if (pCmd[0] == '\0')
    {
        dprintf("lw: Usage: !flcn imemrd <offset> [length(bytes)] [port]\n");
        dprintf("lw: No args specified, defaulted to offset"
                " 0x%04x and length 0x%04x bytes.\n",
                (LwU32)offset, (LwU32)lengthInBytes);
    }

    // Read in the <offset> [length] [port], in that order, if present
    if (GetExpressionEx(pCmd, &offset, &pCmd))
    {
        if (GetExpressionEx(pCmd, &lengthInBytes, &pCmd))
        {
            GetExpressionEx(pCmd, &port, &pCmd);
        }
    }

    // Tidy up the length and offset to be 4-byte aligned
    lengthInBytes   = (lengthInBytes + 3) & ~3ULL;
    offset          = offset & ~3ULL;

    // Get the size of the IMEM and number of IMEM ports
    MGPU_LOOP_START;
    {
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)offset >= memSize)
        {
            dprintf("lw: %s: offset 0x%04x is too large (IMEM size 0x%04x)\n",
                    __FUNCTION__, (LwU32)offset, (LwU32)memSize);
            return;
        }

        // Prevent allocating too much unused memory in temp buffer
        if ((LwU32)(offset + lengthInBytes) >= memSize)
        {
            dprintf("lw: %s: length larger then memory size, truncating to fit\n", __FUNCTION__);
            lengthInBytes = memSize - offset;
        }

        // Create a temporary buffer to store data
        buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
        if (buffer == NULL)
        {
            dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
            return;
        }

        // Actually read the IMEM
        length = pFCIF->flcnImemRead(engineBase,
                                     (LwU32)offset,
                                     (LwU32)lengthInBytes / sizeof(LwU32),
                                     (LwU32)port, buffer);

        // Dump out the IMEM
        if (length > 0)
        {
            dprintf("lw:\tDumping %s IMEM from 0x%04x-0x%04x from port 0x%x:\n",
                    thisFlcn->engineName,
                    (LwU32)offset,
                    (LwU32)(offset + length * sizeof(LwU32)),
                    (LwU32)port);
            printBuffer((char*)buffer, length * sizeof(LwU32), offset, 0x4);
        }

        // Cleanup after ourselves
        free((void*)buffer);
    }
    MGPU_LOOP_END;
}

static void
flcnImemWr
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32  engineBase    = 0x0;
    LwU64  offset        = 0x0;
    LwU64  length        = 0x1;
    LwU64  width         = 0x4;
    LwU64  value         = 0x0;
    LwU64  port          = 0x0;
    LwU32  memSize       = 0x0;
    LwU32  numPorts      = 0x0;
    LwU32  bytesWritten  = 0x0;
    char  *pParams;

    if (pCmd[0] == '\0')
    {
        dprintf("lw: Offset not specified.\n");
        dprintf("lw: Usage: !flcn imemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }

    // extract optional arguments first
    if (parseCmd(pCmd, "w", 1, &pParams))
    {
        GetExpressionEx(pParams, &width, &pParams);
    }

    if (parseCmd(pCmd, "l", 1, &pParams))
    {
        GetExpressionEx(pParams, &length, &pParams);
    }

    if (parseCmd(pCmd, "p", 1, &pParams))
    {
        GetExpressionEx(pParams, &port, &pParams);
    }

    // Read in the <offset> <value>, in that order, if present
    if (!GetExpressionEx(pCmd, &offset, &pCmd))
    {
        dprintf("lw: Value not specified.\n");
        dprintf("lw: Usage: !flcn imemwr <offset> <value> [-w <width>(bytes)] "
                "[-l <length>(units of width)] [-p <port>]\n");
        return;
    }
    GetExpressionEx(pCmd, &value, &pCmd);

    // Get the size of the DMEM and number of DMEM ports
    MGPU_LOOP_START;
    {
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        memSize    = pFCIF->flcnImemGetSize(engineBase);
        numPorts   = pFCIF->flcnImemGetNumPorts(engineBase);

        // Check the port specified
        if (port >= numPorts)
        {
            dprintf("lw: %s: port 0x%x is invalid (max 0x%x)\n",
                    __FUNCTION__, (LwU32)port, (LwU32)(numPorts - 1));
            return;
        }

        // Actually write the IMEM

        dprintf("lw:\tWriting IMEM with pattern 0x%x (width=%d), %d "
                "time(s) starting at IMEM address 0x%0x ...\n",
                (LwU32)value,
                (LwU32)width,
                (LwU32)length,
                (LwU32)offset);

        bytesWritten = pFCIF->flcnImemWrite(
                           engineBase,
                           (LwU32)offset,
                           (LwU32)value,
                           (LwU32)width,
                           (LwU32)length,
                           (LwU32)port);


        dprintf("lw: number of bytes written: 0x%x\n", bytesWritten);
    }
    MGPU_LOOP_END;
}

static void
flcnImblk
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32       engineBase          = 0x0;
    BOOL        bListAll            = TRUE;
    LwU64       userIndex           = ~0x0;
    LwU32       blockIndex          = 0x0;
    LwU32       numBlocks           = 0x0;
    LwU32       tagWidth            = 0x0;
    FLCN_BLOCK  blockInfo;

    // Get the block index
    GetExpressionEx(pCmd, &userIndex, &pCmd);
    bListAll    = (userIndex == (LwU64)(~0));
    blockIndex  = (LwU32)userIndex;

    MGPU_LOOP_START;
    {
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        numBlocks  = pFCIF->flcnImemGetNumBlocks(engineBase);
        tagWidth   = pFCIF->flcnImemGetTagWidth(engineBase);


        if (!bListAll)
        {
            if (blockIndex >= numBlocks)
            {
                dprintf("lw: Block index 0x%x is invalid (max 0x%x).\n",
                        blockIndex, numBlocks - 1);
                return;
            }
            else
            {
                numBlocks = blockIndex + 1;
            }
        }
        else
        {
            blockIndex = 0;
        }

        dprintf("lw:\tDumping IMEM code block status\n");
        dprintf("lw:\t--------------------------------------------------\n");

        // Loop through all the blocks dumping out information
        for (; blockIndex < numBlocks; blockIndex++)
        {
            if (pFCIF->flcnImemBlk(engineBase, blockIndex, &blockInfo))
            {
                dprintf("lw:\tBlock 0x%02x: tag=0x%02x, valid=%d, pending=%d, "
                        "secure=%d\n",
                        blockIndex, blockInfo.tag, blockInfo.bValid,
                        blockInfo.bPending, blockInfo.bSelwre);
            }

        }

        dprintf("\n");
    }
    MGPU_LOOP_END;
}

static void
flcnImtag
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    LwU32       engineBase          = 0x0;
    LwU64       addr                = 0x0;
    LwU32       tagWidth            = 0x0;
    LwU32       maxAddr             = 0x0;
    FLCN_TAG    tagInfo;

    if (pCmd[0] == '\0')
    {
        dprintf("lw:\tUsage: !lw.flcn -<engine> imtag <code addr>\n");
        return;
    }

    // Get the block address to look up
    GetExpressionEx(pCmd, &addr, &pCmd);

    MGPU_LOOP_START;
    {
        //
        // Get the tag width and notify user when code address entered
        // is larger than the maximum allowed by tag width.
        // Note that bits 0-7 are offsets into the block and 8-7+TagWidth
        // define the tag bits.
        //
        pFCIF      = thisFlcn->pFCIF;
        pFEIF      = thisFlcn->pFEIF;
        engineBase = pFEIF->flcnEngGetFalconBase();
        tagWidth   = pFCIF->flcnImemGetTagWidth(engineBase);

        maxAddr = BIT(tagWidth+8) - 1;

        if (addr > maxAddr)
        {
            dprintf("lw: %s: Address 0x%04x is too large (max 0x%04x)\n",
                    __FUNCTION__, (LwU32)addr, maxAddr);
            return;
        }

        dprintf("lw: Dumping block info for address 0x%x, Tag=0x%x\n",
                (LwU32)addr, ((LwU32)addr) >> 8);

        if (pFCIF->flcnImemTag(engineBase, (LwU32)addr, &tagInfo))
        {
            switch (tagInfo.mapType)
            {
            case FALCON_TAG_UNMAPPED:
                dprintf("lw:\tTag 0x%02x: Not mapped to a block", (LwU32)addr >> 8);
                break;
            case FALCON_TAG_MULTI_MAPPED:
            case FALCON_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%d",
                        tagInfo.blockInfo.tag, tagInfo.blockInfo.blockIndex,
                        tagInfo.blockInfo.bValid, tagInfo.blockInfo.bPending,
                        tagInfo.blockInfo.bSelwre);
                break;
            }
            if (tagInfo.mapType == FALCON_TAG_MULTI_MAPPED)
            {
                dprintf(" (multiple)");
            }
            dprintf("\n");
        }
    }
    MGPU_LOOP_END;
}

static void
flcnQueues
(
    char *pCmd
)
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
    const FLCN_ENGINE_IFACES *pFEIF = NULL;
    BOOL       bListAll             = TRUE;
    LwU64      userId               = ~0x0;
    LwU32      queueId              = 0x0;
    LwU32      numQueues            = 0x0;
    FLCN_QUEUE queue;

    GetExpressionEx(pCmd, &userId, &pCmd);
    bListAll = (userId == (LwU64)(~0));

    MGPU_LOOP_START;
    {
        pFCIF     = thisFlcn->pFCIF;
        pFEIF     = thisFlcn->pFEIF;
        numQueues = pFEIF->flcnEngQueueGetNum();

        if (!bListAll)
        {
            if ((LwU32)userId >= numQueues)
            {
                dprintf("lw: %s: 0x%x is not a valid queue ID (max 0x%x)\n",
                        __FUNCTION__, (LwU32)userId, numQueues - 1);
                return;
            }
            queueId   = (LwU32)userId;
            numQueues = queueId + 1;
        }
        else
        {
            queueId = 0;
        }
        for (; queueId < numQueues; queueId++)
        {
            if (pFEIF->flcnEngQueueRead(queueId, &queue))
            {
                flcnQueueDump(FALSE, &queue, thisFlcn->engineName);
            }
        }
        dprintf("\n");
    }
    MGPU_LOOP_END;
}

static void
flcnSt
(
    char *pCmd
)
{
    char * pParam;
    MGPU_LOOP_START;
    {
        FLCN_RTOS_TCB    tcb;
        FLCN_TCB_PVT    *pPrivTcb;
        LwU32            port = 0x1;
        LwU32            ucodeVersion;
        LwU32            pStack = 0;
        LwU32            stackSize = 0;

        // If no permissions to read DMEM, it is not possible to print out anything useful
        if (!flcnDmemRdPermitted())
        {
            return;
        }

        ucodeVersion = thisFlcn->pFCIF->flcnUcodeGetVersion(thisFlcn->engineBase);
        if (parseCmd(pCmd, "l", 1, &pParam))
        {
            flcnstLoad(pParam, ucodeVersion, TRUE);
            return;
        }

        if (parseCmd(pCmd, "u", 0, NULL))
        {
            flcnstUnload();
            return;
        }

        if (flcnRtosTcbGetLwrrent(&tcb, port))
        {
            flcnTcbGetPriv(&pPrivTcb, (LwU32)tcb.flcnTcb.flcnTcb1.pvTcbPvt, port);

            // grab flcn pvt tcb specific variables
            switch (pPrivTcb->tcbPvtVer)
            {
                case FLCN_TCB_PVT_VER_0:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt0.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt0.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_1:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt1.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt1.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_2:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt2.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt2.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_3:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt3.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt3.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_4:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt4.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt4.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_5:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt5.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt5.stackSize;
                    break;

                case FLCN_TCB_PVT_VER_6:
                    pStack    = pPrivTcb->flcnTcbPvt.flcnTcbPvt6.pStack;
                    stackSize = pPrivTcb->flcnTcbPvt.flcnTcbPvt6.stackSize;
                    break;

                default:
                    dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                    break;
            }

            flcnstPrintStacktrace(pStack, stackSize, ucodeVersion);
            free(pPrivTcb);
        }
    }
    MGPU_LOOP_END;
}

static void
flcnTcb
(
    char *pCmd
)
{
    extern POBJFLCN     thisFlcn;
    FLCN_RTOS_TCB       tcb;
    LwU64               tcbAddress = 0x0;
    LwU64               size       = 0x4;
    LwU64               port       = 0x1;
    BOOL                bVerbose   = FALSE;

    CHECK_INIT(MODE_LIVE);

    // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    if ((pCmd == NULL) || (pCmd[0] == '\0'))
    {
        dprintf("lw: Usage: !flcn -<engine> tcb [-v (verbose)] "
            "[-c] [-a] [-l] [tcbAddress] [port] [size].\n");
        return;
    }

    if (parseCmd(pCmd, "v", 0, NULL))
    {
        bVerbose = TRUE;
    }

    // Handle the "-c" case for current TCB.
    if (parseCmd(pCmd, "c", 0, NULL))
    {
        if (GetSafeExpressionEx(pCmd, &port, &pCmd))
        {
            GetSafeExpressionEx(pCmd, &size, &pCmd);
        }
        MGPU_LOOP_START;
        {
            if (thisFlcn && thisFlcn->pFCIF)
            {
                if (!thisFlcn->bSymLoaded)
                {
                    dprintf("lw: symbol not loaded, loading automatically...\n");
                    flcnExec("load", thisFlcn);
                }
                if (flcnRtosTcbGetLwrrent(&tcb, (LwU32)port))
                {
                    dprintf("lw: dumping current TCB\n");
                    dprintf("lw:========================================\n");
                    flcnRtosTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
                }
            }
        }
        MGPU_LOOP_END;
        return;
    }

    // Handle the "-a", or "-l" case for all TCBs
    if ((parseCmd(pCmd, "a", 0, NULL)) ||
        (parseCmd(pCmd, "l", 0, NULL)))
    {
        if (GetSafeExpressionEx(pCmd, &port, &pCmd))
        {
            GetSafeExpressionEx(pCmd, &size, &pCmd);
        }
        MGPU_LOOP_START;
        {
            if (thisFlcn && thisFlcn->pFCIF)
            {
                if (!thisFlcn->bSymLoaded)
                {
                    dprintf("lw: symbol not loaded, loading automatically...\n");
                    flcnExec("load", thisFlcn);
                }
                dprintf("lw: dumping TCBs\n");
                flcnRtosTcbDumpAll(!bVerbose);
            }
        }
        MGPU_LOOP_END;
        return;
    }

    // For a specific tcb address
    if (GetSafeExpressionEx(pCmd, &tcbAddress, &pCmd))
    {
        if (GetSafeExpressionEx(pCmd, &port, &pCmd))
        {
            GetSafeExpressionEx(pCmd, &size, &pCmd);
        }
    }
    MGPU_LOOP_START;
    {
        if (thisFlcn && thisFlcn->pFCIF && (tcbAddress != 0))
        {
            flcnRtosTcbGet((LwU32)tcbAddress, 0, &tcb);
            dprintf("lw: dumping tcb at address 0x%x\n", (LwU32)tcbAddress);
            flcnRtosTcbDump(&tcb, FALSE, (LwU32)port, (LwU8)size);
        }
    }
    MGPU_LOOP_END;
}

static void
flcnEvtq
(
    char *pCmd
)
{
    char    *pParams;
    BOOL     bAll    = FALSE;
    char    *pSym    = NULL;
    LwU64    qAddr   = 0x00;
    LwU64    qTaskId = 0x00;
    LwU64    queueId = LW_U64_MAX;

    CHECK_INIT(MODE_LIVE);

    // If no permissions to read DMEM, it is not possible to print out anything useful
    if (!flcnDmemRdPermitted())
    {
        return;
    }

    if (parseCmd(pCmd, "h", 0, NULL))
    {
        dprintf("lw: Usage: !flcn -<engine> evtq [-h] [-a] [-s <symbol>] [-n <addr>] [-t <task ID>] [-q <queue ID>]\n");
        dprintf("lw:                           - Dump out the DPU RTOS event queues\n");
        dprintf("lw:                           + '-h' : print usage\n");
        dprintf("lw:                           + '-a' : dump all info on all queues\n");
        dprintf("lw:                           + '-s' : dump info on a specific queue (identified by symbol name)\n");
        dprintf("lw:                           + '-n' : dump info on a specific queue (identified by queue address)\n");
        dprintf("lw:                           + '-t' : dump info on a specific queue (identified by task ID)\n");
        dprintf("lw:                           + '-q' : dump info on a specific queue (identified by queue ID)\n");
        return;
    }

    if (parseCmd(pCmd, "a", 0, NULL))
    {
        bAll = TRUE;
    }

    parseCmd(pCmd, "s", 1, &pSym);

    if (parseCmd(pCmd, "n", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &qAddr, &pParams);
    }

    if (parseCmd(pCmd, "t", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &qTaskId, &pParams);
    }

    if (parseCmd(pCmd, "q", 1, &pParams))
    {
        GetSafeExpressionEx(pParams, &queueId, &pParams);
    }

    MGPU_LOOP_START;
    {
        if (thisFlcn && thisFlcn->pFCIF && thisFlcn->pFEIF)
        {
            if (!thisFlcn->bSymLoaded)
            {
                dprintf("lw: symbol not loaded, loading automatically...\n");
                flcnExec("load", thisFlcn);
            }

            if (bAll)
            {
                flcnRtosEventQueueDumpAll(FALSE);
            }
            else if (pSym != NULL)
            {
                flcnRtosEventQueueDumpBySymbol(pSym);
            }
            else if (qAddr != 0)
            {
                flcnRtosEventQueueDumpByAddr((LwU32)qAddr);
            }
            else if (qTaskId != 0)
            {
                flcnRtosEventQueueDumpByTaskId((LwU32)qTaskId);
            }
            else if (queueId != LW_U64_MAX)
            {
                flcnRtosEventQueueDumpByQueueId((LwU32)queueId);
            }
            else
            {
                flcnRtosEventQueueDumpAll(TRUE);
            }
        }
    }
    MGPU_LOOP_END;
}

/*
 * Check if we have permissions to read DMEM
 *
 * @return     LW_TRUE
 *      If allowed to access DMEM
 * @return     LW_FALSE
 *      If not allowed, a hint message will be printed out
 */
static LwBool
flcnDmemRdPermitted(void)
{
    extern POBJFLCN           thisFlcn;
    const FLCN_ENGINE_IFACES *pFEIF      = thisFlcn->pFEIF;
    const FLCN_CORE_IFACES   *pFCIF      = pFEIF->flcnEngGetCoreIFace();
    LwU32                     engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32                     dmemSize   = thisFlcn->pFCIF->flcnDmemGetSize(engineBase);

    if (pFCIF->flcnIsDmemAccessAllowed != NULL)
    {
        if (!pFCIF->flcnIsDmemAccessAllowed(pFEIF,
                                            engineBase,
                                            0,
                                            dmemSize,
                                            LW_TRUE)) // DMEM read
        {
            dprintf("lw: WARNING: DMEM access not permitted, please check if you have psdl license loaded.\n");
            return LW_FALSE;
        }
    }
    return LW_TRUE;
}

