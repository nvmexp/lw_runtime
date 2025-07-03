/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcnsym.c
 * @brief Functions to handle Falcon symbol files parsing
 */

/* ------------------------ Includes --------------------------------------- */
#include "print.h"
#include "falcon.h"

#include "lwsym.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
extern POBJFLCN   thisFlcn;      // from flcndbg.c

/* ------------------------ Static functions ------------------------------- */
void _flcnStrToUpper(char*, const char*);

/* ------------------------ Defines ---------------------------------------- */
#define FLCN_SYM_SIZE_KNOWN(pSym)      ((pSym)->size != 0xffffffff)

void
flcnSymPrintBrief
(
    PFLCN_SYM pSym,
    LwU32     index
)
{
    dprintf("lw:   %6d) %-40s <0x%04x> : (%c) ",
        index, pSym->name, pSym->addr, pSym->section);
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


// all flcnsym-related function require the ability to read files which
// is not possible in the linux kernel mode builds.
void
flcnSymDump
(
    const char *pSymName,
    LwBool      bIgnoreCase
)
{
    PFLCN_SYM  pMatches      = NULL;
    PFLCN_SYM  pSym          = NULL;
    LwU32      count         = 0;
    LwBool     bExactFound   = FALSE;
    LwU32     *pSymValue;
    LwU32      size;
    LwU32      i;

    // Look up the symbol.
    pMatches = flcnSymFind(pSymName, bIgnoreCase, &bExactFound, &count);

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
        flcnSymPrintBrief(pSym, i+1);
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
                wordsRead = thisFlcn->pFCIF->flcnDmemRead(thisFlcn->engineBase,
                                                          pSym->addr,
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

//
// Resolve the given address to a symbol or a list of symbols and dump that
// list out.
//
PFLCN_SYM
flcnSymResolve
(
    LwU32 address
)
{
    PFLCN_SYM pIter;
    PFLCN_SYM pMatches = NULL;

    // Try to auto-load symbols if not lwrrently loaded.
    if (!flcnSymCheckAutoLoad())
    {
        return NULL;
    }

    pIter = thisFlcn->symTable;
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

static LwBool _flcnSymCreate(char *pString, LwU32 tokens, FLCN_SYM **);

void
flcnSymLoad
(
    const char *pFilename,
    LwU32       ucodeVersion
)
{
    PFLCN_SYM pSym = NULL;
    LwU32     index  = 0;
    LwU32     tokens = 1;
    char      line[120];

    LwU8 *pFileBuffer       = NULL;
    LwU32 fileBufferSize    = 0;
    LwU32 fileBufferLwrsor  = 0;
    char *pImplicitFilename = NULL;
    LWSYM_STATUS status;

    static const char *verStr = "AppVersion: ";
    const size_t verLen = strlen(verStr);

    if (thisFlcn->bSymLoaded)
    {
        dprintf("lw: Error: Symbols are already loaded. Please unload first.\n");
        return;
    }

    //
    // If a filename is not specified, load it from the LwSym package
    //
    if ((pFilename == NULL) || (pFilename[0] == '\0'))
    {
        const char *pUcodeName = thisFlcn->pFEIF->flcnEngUcodeName();
        pImplicitFilename = malloc(strlen(thisFlcn->symPath)  +
                                   strlen(pUcodeName)         +
                                   strlen(".nm")              +
                                   1);

        sprintf(pImplicitFilename, "%s%s.nm",
                                   thisFlcn->symPath,
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
                        dprintf("lw: Warning: Falcon ucode version mismatch.\n");
                        dprintf("lw:          on chip: 0x%x, on disk: 0x%x\n",
                                ucodeVersion, version);
                    }
                }
                else
                {
                    dprintf("lw: Error: Unable to determine ucode version on disk.\n");
                }
            }
            else if (_flcnSymCreate(line, tokens, &pSym))
            {
                if (thisFlcn->symTable != NULL)
                {
                    if ((thisFlcn->symTable->bSizeEstimated) &&
                       ((thisFlcn->symTable->section == pSym->section) ||
                       (((thisFlcn->symTable->section == 'b') && (pSym->section == 'B')) ||
                        ((thisFlcn->symTable->section == 'B') && (pSym->section == 'b')) ||
                        ((thisFlcn->symTable->section == 't') && (pSym->section == 'T')) ||
                        ((thisFlcn->symTable->section == 'T') && (pSym->section == 't')))))
                    {
                        thisFlcn->symTable->size = pSym->addr - thisFlcn->symTable->addr;
                    }
                    pSym->pNext = thisFlcn->symTable;
                }
                thisFlcn->symTable = pSym;
            }
            index  = 0;
            tokens = 1;
            dprintf(".");
        }
    }
    dprintf("\n");
    dprintf("lw: %s symbols loaded successfully.\n", thisFlcn->engineName);
    dprintf("lw:\n");
    thisFlcn->bSymLoaded = LW_TRUE;
    lwsymFileUnload(pFileBuffer);
}

void
flcnSymUnload(void)
{
    PFLCN_SYM pSym;
    dprintf("lw: Unloading %s symbols ...\n", thisFlcn->engineName);
    while (thisFlcn->symTable != NULL)
    {
        pSym = thisFlcn->symTable;
        thisFlcn->symTable = thisFlcn->symTable->pNext;
        pSym->pNext = NULL;
        free(pSym);
    }
    thisFlcn->bSymLoaded = LW_FALSE;
    return;
}

/*!
 * Check if the Falcon engine symbols are loaded
 *
 * @return TRUE   loaded
 * @return FALSE  not loaded
 */
LwBool
flcnSymCheckIfLoaded(void)
{
    return thisFlcn->bSymLoaded;
}

/*!
 * @brief Check if symbols are loaded. If they are not, then try to auto-load
 *        them.
 *
 * @return 'TRUE'   If symbols are or were loaded.
 * @return 'FALSE'  If symbols are not and could not be loaded.
 */
LwBool
flcnSymCheckAutoLoad(void)
{
    LwU32 ucodeVersion;
    if (!thisFlcn->bSymLoaded)
    {
        dprintf("lw: Warning: %s symbols not yet loaded. Will attempt "
                "implicit initialization.\n",thisFlcn->engineName);

        ucodeVersion = thisFlcn->pFCIF->flcnUcodeGetVersion(thisFlcn->engineBase);
        flcnSymLoad(NULL, ucodeVersion);
        if (!thisFlcn->bSymLoaded)
        {
            dprintf("lw: Error: %s symbols could not be loaded.\n", thisFlcn->engineName);
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
PFLCN_SYM
flcnSymFind
(
    const char *pSymName,
    LwBool      bIgnoreCase,
    LwBool     *pExactFound,
    LwU32      *pNumFound
)
{
    PFLCN_SYM    pIter         = NULL;
    PFLCN_SYM    pMatches      = NULL;
    PFLCN_SYM    pExact        = NULL;
    char        *pSymNameUpper = NULL;
    const char  *pCmp1         = NULL;
    const char  *pCmp2         = NULL;
    LwU32        count         = 0;

    // Try to auto-load symbols if not lwrrently loaded.
    if (!flcnSymCheckAutoLoad())
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
        _flcnStrToUpper(pSymNameUpper, pSymName);
    }

    // Loop through the symbol table.
    for (pIter = thisFlcn->symTable; pIter != NULL; pIter = pIter->pNext)
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

static LwBool
_flcnSymCreate
(
    char      *pString,
    LwU32      tokens,
    FLCN_SYM **pSym
)
{
    LwU32 result;
    char  section;

    *pSym = (PFLCN_SYM)malloc(sizeof(FLCN_SYM));
    if (*pSym == NULL)
        return LW_FALSE;

    switch (tokens)
    {
        // line for a symbol with a size
        case 4:
        {
            result = sscanf(pString,
                            "%08x %08x %c %40s",
                            &(*pSym)->addr,
                            &(*pSym)->size,
                            &(*pSym)->section,
                             (*pSym)->name);
            if (result != 4)
            {
                free(*pSym); *pSym = NULL;
                return LW_FALSE;
            }
            (*pSym)->bSizeEstimated = LW_FALSE;
            break;
        }
        // line for a symbol without a size
        case 3:
        {
            result = sscanf(pString,
                            "%08x %c %40s",
                            &(*pSym)->addr,
                            &(*pSym)->section,
                             (*pSym)->name);
            if (result != 3)
            {
                free(*pSym); *pSym = NULL;
                return LW_FALSE;
            }
            (*pSym)->size = 0xffffffff;
            (*pSym)->bSizeEstimated = LW_TRUE;
            break;
        }
        // unknown
        default:
        {
            free(*pSym); *pSym = NULL;
            return LW_FALSE;
        }
    }

    section = (*pSym)->section;
    if ((section == 'b') || (section == 'B') ||
        (section == 'd') || (section == 'D') ||
        (section == 'r') || ((*pSym)->addr & 0x10000000))
    {
        (*pSym)->bData = LW_TRUE;
        (*pSym)->addr &= ~0x10000000;
    }
    else
    {
        (*pSym)->bData = LW_FALSE;
    }

    _flcnStrToUpper((*pSym)->nameUpper, (*pSym)->name);
    (*pSym)->pNext  = NULL;
    return LW_TRUE;
}

/*!
 * @brief Colwert a string to upper case.
 *
 * @param[out]  pDst    Destination string.
 * @param[in]   pSrc    Source string.
 */
void
_flcnStrToUpper
(
    char        *pDst,
    const char  *pSrc
)
{
    int i;
    for (i = 0; pSrc[i]; i++)
    {
        pDst[i] = (char)toupper(pSrc[i]);
    }
    pDst[i] = '\0';
}


