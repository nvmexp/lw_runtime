/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmudbg.c
 * @brief WinDbg Extension for PMU.
 *
 *           pmudbg.c
 *              |
 *      +-------+------+
 *      |              |
 *   pmusym.c       pmudt.c
 */

/* ------------------------ Includes --------------------------------------- */
#include "pmu.h"

/* ------------------------ Defines ---------------------------------------- */

#define  PMU_DBG_CHECK_LOADED_OR_BAIL()                                       \
    do                                                                        \
    {                                                                         \
        if (!bLoaded)                                                         \
        {                                                                     \
            dprintf("lw: Error: pmu debug information not loaded. See "       \
                    "'!pmu load'\n");                                         \
            return;                                                           \
        }                                                                     \
    }                                                                         \
    while (0)

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
static BOOL  bSympathSet         = FALSE;
static BOOL  bAutoSympathDerived = FALSE;
static BOOL  bLoaded             = FALSE;
static char  SymPath[256]        = {'\0'};

/* ------------------------ Function Prototypes ---------------------------- */
static char* pmuExecChompWordFromCmd   (char **ppCmd);
static void  pmuExecSympath            (char *pCmd);
static void  pmuExecLoad               (void);
static void  pmuExelwnload             (void);
static BOOL  pmuAutosymDerive          (void);
static void  pmuExecExamineSymbol      (char  *pCmd);
static void  pmuExecDisplayMemory      (char  *pCmd, LwU8 width);
static void  pmuExecImport             (char  *pCmd);
static void  pmuExecRegisterDump       (char  *pCmd);

void
pmuExec
(
    char *pCmd
)
{    
    char *pCmdName;

    pCmdName = pmuExecChompWordFromCmd(&pCmd);
    if (strcmp(pCmdName, "sympath") == 0)
    {
        pmuExecSympath(pCmd);
    }
    else
    if (strcmp(pCmdName, "reload") == 0)
    {
        pmuExelwnload();
        pmuExecLoad();
    }
    else
    if (strcmp(pCmdName, "load") == 0)
    {
        pmuExecLoad();
    }
    else
    if (strcmp(pCmdName, "unload") == 0)
    {
        pmuExelwnload();
    }
    else
    if (strcmp(pCmdName, "dd") == 0)
    {
        pmuExecDisplayMemory(pCmd, 4);
    }
    else
    if (strcmp(pCmdName, "dw") == 0)
    {
        pmuExecDisplayMemory(pCmd, 2);
    }
    else
    if (strcmp(pCmdName, "db") == 0)
    {
        pmuExecDisplayMemory(pCmd, 1);
    }
    else
    if (strcmp(pCmdName, "x") == 0)
    {
        pmuExecExamineSymbol(pCmd);
    }
    else
    if (strcmp(pCmdName, "import") == 0)
    {
        pmuExecImport(pCmd);
    }
    else
    if (strcmp(pCmdName, "r") == 0)
    {
        pmuExecRegisterDump(pCmd);
    }
    else
    {
        dprintf("lw: Unrecognized pmu command: %s\n", pCmdName);
        dprintf("lw:\n");
        dprintf("lw: Available Commands:\n");
        dprintf("lw: dd        dw        db        import\n");
        dprintf("lw: r\n");
    }
}

static char *
pmuExecChompWordFromCmd
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
pmuExecSympath
(
    char   *pCmd
)
{
    char   *pToken;

    pToken = strtok(pCmd, " ");
    if (pToken != NULL)
    {
        memset(SymPath, '\0', sizeof(SymPath));
        strcpy(SymPath, pToken);
    }    
    
    if (*SymPath != '\0')
    {
        dprintf("lw: Symbol search path is: %s\n", SymPath);
        bSympathSet = TRUE;
    }
    else
    if (pmuAutosymDerive())
    {
        bSympathSet = TRUE;
    }
    return;
}

static void
pmuExecLoad(void)
{
    LwU32  ucodeVersion;
    size_t pathLength;
    char  *pFilename;
    char  *pTemp;
    const char *pUcodeName;

    // 
    // If a symbol-path is not set, try to derive one from elw-vars. If that
    // fails, just abort.
    //
    if ((!bSympathSet) && (!pmuAutosymDerive()))
    {
        return;
    }

    // unload any information that's lwrrently loaded
    if (bLoaded)
    {
        pmuExelwnload();
    }

    // determine the version of the ucode/application lwrrently running
    ucodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();

    // 
    // Use the sympath to construct the names of the debug files containing the
    // symbols and types.
    //
    pUcodeName  = pPmu[indexGpu].pmuUcodeName();
    pathLength  = strlen(SymPath)    + 1;
    pathLength += strlen(pUcodeName) + 1;
    pathLength += 2 + 1;

    pFilename = (char *)malloc(pathLength);
    if (pFilename == NULL)
        return;
    strcpy(pFilename, SymPath);
    strcat(pFilename, DIR_SLASH);
    strcat(pFilename, pUcodeName);
    pTemp = pFilename + strlen(pFilename);

    // load the PMU symbols from the nm-file.
    strcpy(pTemp, ".nm");
    pmuSymLoad(pFilename, ucodeVersion);

    free(pFilename);
    bLoaded = TRUE;
}

static void
pmuExelwnload(void)
{
    pmuSymUnload();
    bLoaded = FALSE;
}

static void
pmuExecExamineSymbol
(
    char *pCmd
)
{
    PMU_DBG_CHECK_LOADED_OR_BAIL();
    pmuSymDump(pCmd, TRUE);
}

static BOOL
pmuAutosymDerive(void)
{
    char *pManualDir;

    if (bAutoSympathDerived)
        return TRUE;

    if ((pManualDir = getelw("LWW_MANUAL_SDK")) == NULL)
    {
        PMU_PRINT_SDK_MESSAGE();
        return FALSE;
    }
    memset(SymPath, '\0', sizeof(SymPath));

    strcpy(SymPath, pManualDir);
    strcat(SymPath, DIR_SLASH ".." DIR_SLASH ".." DIR_SLASH ".." DIR_SLASH ".."
                    DIR_SLASH "pmu_sw" DIR_SLASH "prod_app" DIR_SLASH "objdir");

    bAutoSympathDerived = TRUE;
    bSympathSet = TRUE;
    dprintf("lw: Symbol search path is: %s\n", SymPath);
    dprintf("lw:    (derived from LWW_MANUAL_SDK)\n");
    return TRUE;
}

// pmu dd [options] Range
static void
pmuExecDisplayMemory
(
    char  *pCmd,
    LwU8   width
)
{
    LwU64 offset;
    LwU64 lengthInBytes = 0x10;
    LwU64 port = 2;

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
    pmuDmemDump((LwU32)offset, (LwU32)lengthInBytes, (LwU8)port, width);
}

static void
pmuExecRegisterDump
(
    char *pCmd
)
{
    LwU32    val32;
    PMU_SYM *pMatches;
    PMU_SYM *pSym = NULL;
    LwU32    i;
    LwU32    count;

    struct _reg
    {
        LwU32  regIdx;
        char  *pName;
    } regs[] =
    {
        {LW_FALCON_REG_R0   , "r0"  },
        {LW_FALCON_REG_R1   , "r1"  },
        {LW_FALCON_REG_R2   , "r2"  },
        {LW_FALCON_REG_R3   , "r3"  },
        {LW_FALCON_REG_R4   , "r4"  },
        {LW_FALCON_REG_R5   , "r5"  },
        {LW_FALCON_REG_R6   , "r6"  },
        {LW_FALCON_REG_R7   , "r7"  },
        {LW_FALCON_REG_R8   , "r8"  },
        {LW_FALCON_REG_R9   , "r9"  },
        {LW_FALCON_REG_R10  , "r10" },
        {LW_FALCON_REG_R11  , "r11" },
        {LW_FALCON_REG_R12  , "r12" },
        {LW_FALCON_REG_R13  , "r13" },
        {LW_FALCON_REG_R14  , "r14" },
        {LW_FALCON_REG_R15  , "r15" },
        {LW_FALCON_REG_IV0  , "iv0" },
        {LW_FALCON_REG_IV1  , "iv1" },
        {LW_FALCON_REG_EV   , "ev"  },
        {LW_FALCON_REG_IMB  , "imb" },
        {LW_FALCON_REG_DMB  , "dmb" },
        {LW_FALCON_REG_CSW  , "csw" },
        {LW_FALCON_REG_CCR  , "ccr" },
        {LW_FALCON_REG_SEC  , "sec" },
        {LW_FALCON_REG_CTX  , "ctx" },
        {LW_FALCON_REG_EXCI , "exci"},
    };

    dprintf("lw: Dumping PMU Falcon Registers...\n");
    dprintf("lw:\n");

    // 
    // Dump the PC first including the nearest code symbol name (if symbols are
    // loaded).
    //
    val32 = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_PC);
    dprintf("lw:   pc=0x%08x", val32);
    if ((val32 != 0xFFFFFFFF) && pmuSymCheckIfLoaded())
    {
        pMatches = pmuSymResolve(val32);
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
    val32 = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_SP);
    dprintf("lw:   sp=0x%08x\n", val32);
    dprintf("lw:\n");
    dprintf("lw: ");

    count = sizeof(regs) / sizeof(struct _reg);
    for (i = 0; i < count; i++)
    {
        val32 = pPmu[indexGpu].pmuFalconGetRegister(regs[i].regIdx);
        dprintf("%4s=0x%08x", regs[i].pName, val32);
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
pmuExecImport
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
        dprintf("lw: usage: !pmu import <offset> <filename>\n");
        return;
    }
    start = (LwU32)offset;

    pFilename = strtok(pCmd, " ");
    if (pFilename == NULL)
    {
        dprintf("lw: import filename not given\n");
        dprintf("lw: usage: !pmu import <offset> <filename>\n");
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
            if (pPmu[indexGpu].pmuDmemWrite((LwU32)offset,
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
        dprintf("lw: Imported %d bytes to 0x%04x-0x%04x\n",
            c, start, start + c - 1);
    }
    else
    {
        dprintf("lw: FAILED to write to DMEM offset 0x%x",
                (LwU32)offset);
    }
    dprintf("lw:\n");
}
