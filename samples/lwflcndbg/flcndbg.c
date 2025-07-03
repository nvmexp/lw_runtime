/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
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
#include "hal.h"
#include "falcon.h"

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

#define FLCN_PRINT_USAGE_MESSAGE()                                                  \
    dprintf("lw: Avaiable commands:\n");                                            \
    dprintf("lw: dd        dw        db        r       import\n");                  \
    dprintf("lw: sympath   load      unload    x\n");

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
POBJFLCN thisFlcn = NULL;

/* ------------------------ Function Prototypes ---------------------------- */
static char*   flcnExecChompWordFromCmd   (char **ppCmd);
static void    flcnExecSympath            (char *pCmd);
static void    flcnExecLoad               (void);
static void    flcnExelwnload             (void);
static LwBool  flcnAutosymDerive          (void);
static void    flcnExecExamineSymbol      (char  *pCmd);
static void    flcnExecDisplayMemory      (char  *pCmd, LwU8 width);
static void    flcnExecImport             (char  *pCmd);
static void    flcnExecRegisterDump       (char  *pCmd);

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
        flcnExecLoad();
    }
    else if (strcmp(pCmdName, "reload") == 0)
    {
        flcnExelwnload();
        flcnExecLoad();
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
    else
    {
        dprintf("lw: Unrecognized flcn command: %s\n", pCmdName);
        dprintf("lw:\n");
        FLCN_PRINT_USAGE_MESSAGE();
    }
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
        return TRUE;
    
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
flcnExecLoad(void)
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

    // determine the version of the ucode/application lwrrently running

    ucodeVersion = thisFlcn->pFCIF->flcnUcodeGetVersion(thisFlcn->engineBase);
    pUcodeName   = thisFlcn->pFEIF->flcnEngUcodeName();
    
    pathLength  = strlen(thisFlcn->symPath) + 1;
    pathLength += strlen(pUcodeName) + 1;
    pathLength += 2 + 1;

    pFilename = (char *)malloc(pathLength);
    if (pFilename == NULL)
        return;
    strcpy(pFilename, thisFlcn->symPath);
    strcat(pFilename, DIR_SLASH);
    strcat(pFilename, pUcodeName);
    pTemp = pFilename + strlen(pFilename);

    // load the FALCON symbols from the nm-file.
    strcpy(pTemp, ".nm");
    flcnSymLoad(pFilename, ucodeVersion);

    free(pFilename);
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
    LwU64 port = 2;

    if (!thisFlcn->pFEIF)
    {
        FLCN_PRINT_NOTARGET_MESSAGE();
        return;
    }

    if (!GetExpressionEx(pCmd, &offset, &pCmd))
        return;

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
            thisFlcn->pFCIF->flcnDmemWrite(thisFlcn->engineBase, 
                                              (LwU32)offset, 
                                              w, 0x4, 0x1, 0x1);
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
    dprintf("lw: OK\n");
    dprintf("lw: Imported %d bytes to 0x%04x-0x%04x %s DMEM\n",
        c, start, start + c - 1, thisFlcn->engineName);
    dprintf("lw:\n");
}
