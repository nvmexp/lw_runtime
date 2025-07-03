/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmusym.c
 * @brief WinDbg Extension for PMU.
 */

/* ------------------------ Includes --------------------------------------- */
#include "pmu.h"
#include "print.h"

#include "lwsym.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
static PMU_SYM *PmuSymTable    = NULL;
static char    *pManualDir     = NULL;
static BOOL     bSymbolsLoaded = FALSE;

/* ------------------------ Function Prototypes ---------------------------- */
/* ------------------------ Defines ---------------------------------------- */
#define PMU_SYM_SIZE_KNOWN(pSym)      ((pSym)->size != 0xffffffff)

#define LW_FLCN_DMTAG_BLK_LINE_INDEX                                        7:0

#define DMEM_BLOCK_WIDTH            8
#define DMEM_BLOCK_SIZE             BIT(DMEM_BLOCK_WIDTH)
#define DMEM_IDX_TO_ADDR(i)         (((LwU32)(i)) << DMEM_BLOCK_WIDTH)
#define DMTAG_IDX_GET(s)            DRF_VAL(_FLCN, _DMTAG, _BLK_LINE_INDEX, (s))

void
pmuSymPrintBrief
(
    PMU_SYM *pSym,
    LwU32    index
)
{
    char fmt[64] = {0};

    sprintf(fmt, "%s%d%s", "lw:   %6d) %-", PMU_SYM_NAME_SIZE, "s <0x%04x> : (%c) ");
    dprintf(fmt, index, pSym->name, pSym->addr, pSym->section);
    if (pSym->section != 'A')
    {
        dprintf("%s ", pSym->bData ? "DATA" : "CODE");
        if (pSym->size == 0xffffffff)
        {
            dprintf(": unknown size\n");
        }
        else
        {
            dprintf(": size=%-4d (0x%04x)", pSym->size, pSym->size);
            if (pSym->bSizeEstimated)
            {
                dprintf(" - estimated");
            }
            dprintf("\n");
        }
    }
    else
    {
        dprintf("ABS\n");
    }
}

static BOOL _pmuSymCreate(char *pString, LwU32 tokens, PMU_SYM **);

void
pmuSymLoad
(
    const char *pFilename,
    LwU32       ucodeVersion
)
{
    PMU_SYM *pSym = NULL;
    LwU32    index  = 0;
    LwU32    tokens = 1;
    char     line[120];

    LwU8 *pFileBuffer       = NULL;
    LwU32 fileBufferSize    = 0;
    LwU32 fileBufferLwrsor  = 0;
    char *pImplicitFilename = NULL;
    LWSYM_STATUS status;

    static const char *verStr = "AppVersion: ";
    const size_t verLen = strlen(verStr);

    if (bSymbolsLoaded)
    {
        dprintf("lw: Error: Symbols are already loaded. Please unload "
                "first using \"pmusym -u\".\n");
        return;
    }

    //
    // If a filename is not specified, load it from the LwSym package
    //
    if ((pFilename == NULL) || (pFilename[0] == '\0'))
    {
        const char *pUcodeName = pPmu[indexGpu].pmuUcodeName();
        pImplicitFilename = malloc(strlen(LWSYM_VIRUTAL_PATH) +
                                   strlen("pmusym/")          +
                                   strlen(pUcodeName)         +
                                   strlen(".nm")              +
                                   1);

        sprintf(pImplicitFilename, "%s%s%s.nm",
                                   LWSYM_VIRUTAL_PATH,
                                   "pmusym/",
                                   pUcodeName);
        pFilename = pImplicitFilename;
    }


    dprintf("lw: Loading symbol file: %s\n", pFilename);
    status = lwsymFileLoad(pFilename, &pFileBuffer, &fileBufferSize);
    if (pImplicitFilename != NULL)
    {
        free(pImplicitFilename);
        pImplicitFilename = NULL;
        pFilename = NULL;
    }

    if (status != LWSYM_OK)
    {
        dprintf("lw: Error reading file (%s)\n", lwsymGetStatusMessage(status));
        return;
    }


    index = 0;
    //
    // Now read the file in character by character.  Keep track of newlines
    // and whitespace (to count tokens). Each line will represent a symbol.
    // The token count is needed since not all symbols will be listed with
    // their size and the token count is the easiest way to detect those cases.
    //
    while (fileBufferLwrsor < fileBufferSize)
    {
        line[index] = pFileBuffer[fileBufferLwrsor++];

        if (line[index] != '\n')
        {
            if (line[index] == ' ')
            {
                tokens++;
            }
            index++;
        }
        else
        {
            line[index+1] = '\0';

            //
            // If the line starts with "AppVersion: ", compare it to the
            // requested version.  We need to specify the length exactly so
            // that it still matches although "line" doesn't have a NUL
            // terminator there.
            //
            if (!strncmp(line, verStr, verLen))
            {
                LwU32 version = 0;
                int ret = sscanf(line + verLen, "%u", &version);
                if (ret && ret != EOF)
                {
                    if (version != ucodeVersion)
                    {
                        dprintf("lw: Warning: PMU ucode version mismatch.\n");
                        dprintf("lw:          on chip: %d, on disk: %d\n",
                                ucodeVersion, version);
                    }
                }
                else
                {
                    dprintf("lw: Error: Unable to determine ucode version on disk.\n");
                }
            }
            else if (_pmuSymCreate(line, tokens, &pSym))
            {
                if (PmuSymTable != NULL)
                {
                    if ((PmuSymTable->bSizeEstimated) &&
                       ((PmuSymTable->section == pSym->section) ||
                       (((PmuSymTable->section == 'b') && (pSym->section == 'B')) ||
                        ((PmuSymTable->section == 'B') && (pSym->section == 'b')) ||
                        ((PmuSymTable->section == 't') && (pSym->section == 'T')) ||
                        ((PmuSymTable->section == 'T') && (pSym->section == 't')))))
                    {
                        PmuSymTable->size = pSym->addr - PmuSymTable->addr;
                    }
                    pSym->pNext = PmuSymTable;
                }
                PmuSymTable = pSym;
            }
            index  = 0;
            tokens = 1;
            dprintf(".");
        }
    }
    dprintf("\n");
    dprintf("lw: PMU symbols loaded successfully.\n");
    dprintf("lw:\n");
    bSymbolsLoaded = TRUE;

    lwsymFileUnload(pFileBuffer);

    return;
}

void
pmuSymUnload(void)
{
    PMU_SYM *pSym;
    dprintf("lw: Unloading PMU symbols ...\n");
    while (PmuSymTable != NULL)
    {
        pSym = PmuSymTable;
        PmuSymTable = PmuSymTable->pNext;
        pSym->pNext = NULL;
        free(pSym);
    }
    pManualDir     = NULL;
    bSymbolsLoaded = FALSE;
    return;
}

void
pmuSymDump
(
    const char *pSymName,
    BOOL        bIgnoreCase
)
{
    PMU_SYM *pMatches      = NULL;
    PMU_SYM *pSym          = NULL;
    LwU32    count         = 0;
    BOOL     bExactFound   = FALSE;
    LwU32   *pSymValue;
    LwU32    size;
    LwU32    i;

    // Look up the symbol.
    pMatches = pmuSymFind(pSymName, bIgnoreCase, &bExactFound, &count);

    // Print out the number of matches found.
    dprintf("lw: Looking up symbol \"%s\" ... ", pSymName);
    switch (count)
    {
        case 0:
            dprintf("no matches found!\n");
            return;
        case 1:
            dprintf("1 match found.\n");
            break;
        default:
            dprintf("%d matches found!\n", count);
            break;
    }

    // Print out all the matches found.
    for (i = 0, pSym = pMatches; pSym != NULL; pSym = pSym->pTemp, i++)
    {
        pmuSymPrintBrief(pSym, i+1);
    }
    dprintf("lw:\n");

    // Print out details if exact match, or only 1 match found.
    if (bExactFound || count == 1)
    {
        // First symbol in list will be the exact match symbol.
        pSym = pMatches;

        if (pSym->size != 0xffffffff)
        {
            dprintf("lw: Dumping value of symbol <%s>\n", pSym->name);
            size = (pSym->size >= 4) ? pSym->size : 4;
        }
        else
        {
            dprintf("lw: Size of symbol <%s> is unknown, dumping first 4-bytes "
                    "only.\n", pSym->name);
            size = 0x4;
        }

        pSymValue = (LwU32 *)malloc(size);
        if (pSymValue != NULL)
        {
            if (pSym->bData)
            {
                LwU32 wordsRead;
                wordsRead = pPmu[indexGpu].pmuDmemRead(pSym->addr,
                                                       LW_TRUE,
                                                       size >> 2,
                                                       1,
                                                       pSymValue);
                if (wordsRead == (size >> 2))
                {
                    printBuffer((char*)pSymValue, size, pSym->addr, 4);
                }
                else
                {
                    dprintf("lw: Unable to read data at address 0x%x. "
                            "Cannot dump <%s>\n", pSym->addr, pSym->name);
                }
            }
            else
            {
                // code-symbol dumping not supported
                dprintf("lw: Dumping symbols in CODE section not supported. "
                        "Cannot dump <%s>\n", pSym->name);
            }
            free(pSymValue);
            pSymValue = NULL;
        }
    }
    return;
}

/*!
 * @brief  Load a PMU file with a given extension from lwsym into a provided buffer.
 *
 * @param[in]  pFileExtension   Expected extension of the file to load, needs to start with .
 * @param[out] ppFileBuffer     Pointer to buffer storing the file data
 * @param[in]  pFileBufferSize  Pointer to variable holding the buffer size
 * @param[in]  bVerbose         Print out all the intermediate parsing info or not
 *
 * @return     TRUE
 *      Load succeeded
 *             FALSE
 *      Load failed
 */
BOOL pmuSymLwsymFileLoad
(
    const char  *pFileExtension,
    LwU8       **ppFileBuffer,
    LwU32       *pFileBufferSize,
    BOOL         bVerbose
)
{
    LWSYM_STATUS status;

    char        *pLwsymFilename = NULL;
    const char  *pUcodeName     = pPmu[indexGpu].pmuUcodeName();

    if ((pFileExtension == NULL) || (pFileExtension[0] != '.'))
    {
        dprintf("lw: Error reading file! Expected valid extension starting with a .\n");
    }

    pLwsymFilename = malloc(strlen(LWSYM_VIRUTAL_PATH) +
                            strlen("pmusym/")          +
                            strlen(pUcodeName)         +
                            strlen(pFileExtension)     +
                            (size_t)1);

    if (pLwsymFilename == NULL)
    {
        dprintf("lw: Error reading file! (Out of memory)\n");
        return FALSE;
    }

    sprintf(pLwsymFilename, "%s%s%s%s",
                            LWSYM_VIRUTAL_PATH,
                            "pmusym/",
                            pUcodeName,
                            pFileExtension);

    if (bVerbose)
    {
        dprintf("lw: Loading symbol file: %s\n", pLwsymFilename);
    }
    status = lwsymFileLoad(pLwsymFilename, ppFileBuffer, pFileBufferSize);
    free(pLwsymFilename);

    if (status != LWSYM_OK)
    {
        dprintf("lw: Error reading file (%s)\n", lwsymGetStatusMessage(status));
        return FALSE;
    }

    return TRUE;
}

//
// Resolve the given address to a symbol or a list of symbols and dump that
// list out.
//
PMU_SYM *
pmuSymResolve
(
    LwU32 address
)
{
    PMU_SYM *pIter;
    PMU_SYM *pMatches = NULL;

    // Try to auto-load symbols if not lwrrently loaded.
    if (!pmuSymCheckAutoLoad())
    {
        return NULL;
    }

    pIter = PmuSymTable;
    for (;pIter != NULL; pIter = pIter->pNext)
    {
        if (pIter->size == 0xFFFFFFFF)
            continue;

        if ((pIter->addr <= address) && (address < (pIter->addr + pIter->size)))
        {
            pIter->pTemp = pMatches;
            pMatches     = pIter;
        }
    }
    return pMatches;
}

/*!
 * Check if the PMU symbols are loaded
 *
 * @return TRUE   loaded
 * @return FALSE  not loaded
 */
BOOL
pmuSymCheckIfLoaded(void)
{
    return bSymbolsLoaded;
}


/*!
 * @brief Check if symbols are loaded. If they are not, then try to auto-load
 *        them.
 *
 * @return 'TRUE'   If symbols are or were loaded.
 * @return 'FALSE'  If symbols are not and could not be loaded.
 */
BOOL
pmuSymCheckAutoLoad(void)
{
    LwU32 ucodeVersion;
    if (!bSymbolsLoaded)
    {
        dprintf("lw: Warning: PMU symbols not yet loaded. Will attempt "
                "implicit initialization.\n");

        ucodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();
        pmuSymLoad(NULL, ucodeVersion);
        if (!bSymbolsLoaded)
        {
            dprintf("lw: Error: PMU symbols could not be loaded.\n");
            return FALSE;
        }
    }
    return TRUE;
}

/*!
 * @brief Look for matching symbols in the symbol table.
 *
 * Exact matches ignore the leading underscore in symbol names.
 *
 * @param[in]   pSymName    Name of symbol to search for.
 * @param[in]   bIgnoreCase Whether or not to ignore case.
 * @param[out]  pExactFound Whether an exact symbol match was found.
 * @param[out]  pNumFound   Number of matches found (includes exact match).
 *
 * @return Pointer to the temporary linked list of symbols. If an exact match
 *         was found, then the exact match is the first element on the list.
 */
PMU_SYM *
pmuSymFind
(
    const char *pSymName,
    BOOL        bIgnoreCase,
    BOOL       *pExactFound,
    LwU32      *pNumFound
)
{
    PMU_SYM    *pIter         = NULL;
    PMU_SYM    *pMatches      = NULL;
    PMU_SYM    *pExact        = NULL;
    char       *pSymNameUpper = NULL;
    const char *pCmp1         = NULL;
    const char *pCmp2         = NULL;
    LwU32       count         = 0;

    // Try to auto-load symbols if not lwrrently loaded.
    if (!pmuSymCheckAutoLoad())
    {
        return NULL;
    }

    // If we are ignoring case, colwert name to upper case.
    if (bIgnoreCase)
    {
        pSymNameUpper = (char *)malloc(strlen(pSymName) + 1);
        if (pSymNameUpper == NULL)
        {
            dprintf("lw: %s: Error: Could not allocate buffer!\n", __FUNCTION__);
            return NULL;
        }
        pmuStrToUpper(pSymNameUpper, pSymName);
    }

    // Loop through the symbol table.
    for (pIter = PmuSymTable; pIter != NULL; pIter = pIter->pNext)
    {
        // Get the strings we want to compare.
        pCmp1 = bIgnoreCase ? pSymNameUpper    : pSymName;
        pCmp2 = bIgnoreCase ? pIter->nameUpper : pIter->name;

        // Ignore the leading '_' character in symbol table for exact match.
        if (strcmp(pCmp1, pCmp2+1) == 0)
        {
            pExact = pIter;
        }
        else if (strstr(pCmp2, pCmp1) != NULL)
        {
            // Add the symbol to the front of a temporary linked list.
            pIter->pTemp = pMatches;
            pMatches     = pIter;
            count++;
        }
    }

    // If we found an exact match, put it at the front of the list.
    if (pExact != NULL)
    {
        pExact->pTemp = pMatches;
        pMatches      = pExact;
        count += 1;
    }

    // Free the upper case string if allocated one.
    if (pSymNameUpper != NULL)
    {
        free(pSymNameUpper);
    }

    // Write output parameters.
    *pExactFound = pExact != NULL;
    *pNumFound   = count;
    return pMatches;
}

static BOOL
_pmuSymCreate
(
    char     *pString,
    LwU32     tokens,
    PMU_SYM **pSym
)
{
    LwU32 result;
    char  section;

    *pSym = (PMU_SYM*)malloc(sizeof(PMU_SYM));
    if (*pSym == NULL)
        return FALSE;

    switch (tokens)
    {
        // line for a symbol with a size
        case 4:
        {
            char sscanf_fmt[24] = {0};
            sprintf(sscanf_fmt, "%s%d%s", "%08x %08x %c %", PMU_SYM_NAME_SIZE, "s");
            result = sscanf(pString,
                            sscanf_fmt,
                            &(*pSym)->addr,
                            &(*pSym)->size,
                            &(*pSym)->section,
                             (*pSym)->name);
            if (result != 4)
            {
                free(*pSym); *pSym = NULL;
                return FALSE;
            }
            (*pSym)->bSizeEstimated = FALSE;
            break;
        }
        // line for a symbol without a size
        case 3:
        {
            char sscanf_fmt[16] = {0};
            sprintf(sscanf_fmt, "%s%d%s", "%08x %c %", PMU_SYM_NAME_SIZE, "s");
            result = sscanf(pString,
                            sscanf_fmt,
                            &(*pSym)->addr,
                            &(*pSym)->section,
                             (*pSym)->name);
            if (result != 3)
            {
                free(*pSym); *pSym = NULL;
                return FALSE;
            }
            (*pSym)->size = 0xffffffff;
            (*pSym)->bSizeEstimated = TRUE;
            break;
        }
        // unknown
        default:
        {
            free(*pSym); *pSym = NULL;
            return FALSE;
        }
    }

    section = (*pSym)->section;
    if ((section == 'b') || (section == 'B') ||
        (section == 'd') || (section == 'D') ||
        (section == 'r') || ((*pSym)->addr & 0x10000000))
    {
        (*pSym)->bData = TRUE;
        (*pSym)->addr &= ~0x10000000;
    }
    else
    {
        (*pSym)->bData = FALSE;
    }

    pmuStrToUpper((*pSym)->nameUpper, (*pSym)->name);
    (*pSym)->pNext  = NULL;
    return TRUE;
}
