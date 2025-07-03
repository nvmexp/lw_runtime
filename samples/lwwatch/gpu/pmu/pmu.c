/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmu.c
 * @brief WinDbg Extension for PMU.
 */

/* ------------------------ Includes --------------------------------------- */
#include "pmu.h"
#include "print.h"
#include "lwsym.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
static OBJFLCN pmuFlcn = {0};

/* ------------------------ Function Prototypes ---------------------------- */
/* ------------------------ Defines ---------------------------------------- */

void
pmuDmemDump
(
    LwU32 offset,
    LwU32 lengthInBytes,
    LwU8  port,
    LwU8  size
)
{
    LwU32   memSize  = 0x0;
    LwU32   numPorts = 0x0;
    LwU32   length   = 0x0;
    LwU32*  buffer   = NULL;

    // Tidy up the length to be 4-byte aligned
    lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    offset = offset & ~3ULL;

    // Get the size of the DMEM and number of DMEM ports
    memSize  = pPmu[indexGpu].pmuDmemGetSize();
    numPorts = pPmu[indexGpu].pmuDmemGetNumPorts();

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
        dprintf("lw: %s: offset 0x%x is too large (DMEM size 0x%x)\n",
                __FUNCTION__, (LwU32)offset, (LwU32)memSize);
        return;
    }

    // Prevent allocating too much unused memory in temp buffer
    if ((LwU32)(offset + lengthInBytes) >= memSize)
    {
        dprintf("lw: %s: length larger then memory size, truncating to fit\n", 
                __FUNCTION__);
        lengthInBytes = memSize - offset;
    }
    
    // Create a temporary buffer to store data
    buffer = (LwU32 *)malloc((LwU32)lengthInBytes);
    if (buffer == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return;
    }

    // Actually read the DMEM
    length = pPmu[indexGpu].pmuDmemRead((LwU32)offset, 
                                LW_FALSE,
                                (LwU32)lengthInBytes / sizeof(LwU32), 
                                (LwU32)port, buffer);

    // Dump out the DMEM
    if (length > 0)
    {
        dprintf("lw:\tDumping DMEM from 0x%04x-0x%04x from port 0x%x:\n",
                (LwU32)offset, 
                (LwU32)(offset + length * sizeof(LwU32)), 
                (LwU32)port);
        printBuffer((char*)buffer, length * sizeof(LwU32), offset, (LwU8)size);
    }

    // Cleanup after ourselves
    free(buffer);
    return;
}

/*!
 * For each code tag between minTag and maxTag, check to see if its
 * loaded/mapped into IMEM. Dump the reults in an easy to read table.
 * 
 * @param[in]  minTag  Code tag to start the dump with
 * @param[in]  maxTag  Code tag to end the dump with
 * @param[in]  bSkipUnmappedTags
 *    TRUE if nothing should be printed for any tags that are not mapped
 */
void
pmuImemMapDump
(
    LwU32 minTag,
    LwU32 maxTag,
    BOOL bSkipUnmappedTags
)
{
    LwU32       tag = 0x0;
    PmuTagBlock tagMapping;
    LwU32       bIncludeSymbols;
    PMU_SYM    *pMatches;
    PMU_SYM    *pSym = NULL;
    
    bIncludeSymbols = pmuSymCheckIfLoaded();

    // Print out a pretty little header
    dprintf("lw:\tDumping IMEM tag to block mapping\n");
    dprintf("lw:\t----------------------------------------------------------------------------------\n");

    // Now we know how many tags to look through
    for (tag = minTag; tag <= maxTag; tag++)
    {
        pPmu[indexGpu].pmuImtag((tag << 8), &tagMapping);
        switch (tagMapping.mapType)
        {
            case PMU_TAG_UNMAPPED: 
                if (!bSkipUnmappedTags)
                {
                    dprintf("lw:\tTag 0x%02x: Not mapped to a block\n", tag);
                }
                break;

            case PMU_TAG_MULTI_MAPPED:
            case PMU_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%-1d",
                        tag, tagMapping.blockInfo.blockIndex, tagMapping.blockInfo.bValid,
                        tagMapping.blockInfo.bPending, tagMapping.blockInfo.bSelwre);

                if (bIncludeSymbols)
                {
                    pMatches = pmuSymResolve(tag * 256);
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
                        dprintf(" : ");
                        if (pSym->name[1] != '_')
                        {
                            dprintf(" ");
                        }
                        dprintf("%s", pSym->name);
                    }
                }

                if (tagMapping.mapType == PMU_TAG_MULTI_MAPPED)
                {
                    dprintf(" (multiple)");
                }
                dprintf("\n");
                break;
        }
    }

    if (!bIncludeSymbols)
    {
        dprintf("\nlw:\tTo dump the nearest CODE symbol for each tag address, you need"
                "to load the PMU symbols using '!pmusym -l' first\n");
    }
}

void
pmuSimpleBootstrap
(
    const char *pFilename
)
{
    LwU32   csize, dsize, totalsize;
    LwU32   i, blockCount, wordCount, byteCount;
    size_t  nread = 0;
    FILE   *pFile;
    char    line[1024];
    LwU8   *pBin = NULL;
    LwU32   bytesWritten;
    LwBool  bWriteSuccess = LW_TRUE;

    pFile = fopen(pFilename, "rb");
    if (pFile == NULL)
    {
        dprintf("lw: Error: cannot open PMU program: %s\n", pFilename);
        return;
    }

    // Read PMU binary signature.
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open PMU program: %s\n", pFilename);
        return;
    }
    nread = strlen(line);
    line[nread - 1] = '\0';
    if (strcmp(line, "PMU_"))
    {
        dprintf("lw: Error: Invalid PMU Simple Program: %s\n", pFilename);
        return;
    }
    // Read Code size 
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open PMU program: %s\n", pFilename);
        return;
    }
    if (sscanf(line, "%d\n", &csize) != 1)
    {
        dprintf("lw: Error: Invalid PMU Simple Program: %s\n", pFilename);
        return;
    }

    // Read Data size 
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open PMU program: %s\n", pFilename);
        return;
    }
    if (sscanf(line, "%d\n", &dsize) != 1)
    {
        dprintf("lw: Error: Invalid PMU Simple Program: %s\n", pFilename);
        return;
    }

    totalsize = csize + dsize;
    
    if ((pBin = (LwU8 *) malloc(totalsize)) == NULL)
    {
        dprintf("lw: Error: Could not allocate memory.\n"); 
        return;
    }

    // read the binary
    if ((nread = fread(pBin, totalsize, 1, pFile)) != 1)
    {
        dprintf("lw: Error : Could not read the binary"
                " (%d/%d read, code = %d, data = %d).\n", 
                (LwU32)nread, totalsize, csize, dsize); 
        free(pBin);
        return;
    }

    // reset PMU
    pPmu[indexGpu].pmuMasterReset();

    dprintf("lw: Loading ucode....[-]\r");
    //
    // Bootstrap
    //

    // 1. copy code to IMEM
    wordCount = csize / 4; 
    for (i = 0; i < wordCount; i++)
    {
        // check for beginning of a block.
        if ((i % (256/4)) == 0)
        {
            pPmu[indexGpu].pmuImemSetTag((i * 4) >> 8, 0); 
        }

        pPmu[indexGpu].pmuImemWrite(i*4, ((LwU32 *)pBin)[i], 4, 1, 0); 
        dprintf("lw: Loading ucode....[%c]\r", "\\|/-"[i%4]);
        fflush(stdout);
    }
    // copy the rest
    byteCount = csize % 4;
    if (byteCount)
    {
        if ((wordCount % (256/4)) == 0)
        {
            pPmu[indexGpu].pmuImemSetTag((wordCount * 4) >> 8, 0); 
        }
        for (i = 0 ; i < byteCount ; i++)
        {
            pPmu[indexGpu].pmuImemWrite(wordCount * 4 + i, pBin[(wordCount*4) + i], 1, 1, 0);
        }
    }
    
    // write dummy data at the end of IMEM block to mark the last block 'valid'
    if (csize % 256)
    {
        blockCount = csize/256 + 1;
        //                           addr, data, width, size, port
        pPmu[indexGpu].pmuImemWrite(blockCount * 256 - 4, 0, 4, 1, 0);
    }


    // 2. copy data to DMEM
    // copy by words first 
    wordCount = dsize / 4; 
    for (i = 0; i < wordCount; i++)
    {
        bytesWritten = pPmu[indexGpu].pmuDmemWrite(i*4, LW_FALSE,
                                                   ((LwU32 *)(pBin+csize))[i], 4, 1, 0);
        if (bytesWritten != 4)
        {
            dprintf("lw: Failed to write at offset 0x%x\n", i * 4);
            bWriteSuccess = LW_FALSE;
            break;
        }
        dprintf("lw: Loading ucode....[%c]\r", "\\|/-"[i%4]);
        fflush(stdout);
    }
    // copy the rest
    if (bWriteSuccess)
    {
        byteCount = dsize % 4;
        for (i = 0 ; i < byteCount ; i++)
        {
            bytesWritten = pPmu[indexGpu].pmuDmemWrite(wordCount * 4,
                                                       LW_FALSE,
                                                       (pBin + csize)[(wordCount*4) + i],
                                                       1, 1, 0);
            if (bytesWritten != 1)
            {
                dprintf("lw: Failed to write at offset 0x%x\n", wordCount * 4);
                bWriteSuccess = LW_FALSE;
                break;
            }
        }
    }
    // free memory
    free(pBin);
    fclose(pFile);

    if (bWriteSuccess)
    {
        // start at 0
        pPmu[indexGpu].pmuBootstrap(0);

        dprintf("lw:\nlw: Starting %s!\n", pFilename);
    }
    else
    {
        dprintf("lw:\nlw: Cannot start %s!\n", pFilename);
    }
}

/*!
 * @brief Colwert a string to upper case.
 *
 * @param[out]  pDst    Destination string.
 * @param[in]   pSrc    Source string.
 */
void
pmuStrToUpper
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

void
pmuStrLwtRegion
(
    char   **ppStr,
    LwU32    offs,
    LwU32    len
)
{
    char *pEnd;
    char *pTemp;

    if (offs == 0)
    {
        *ppStr += len;
    }

    pEnd  = *ppStr + offs;
    pTemp = pEnd + len;

    while (*pTemp != '\0')
    {
        *pEnd = *pTemp;
        pEnd++;
        pTemp++;
    }
    *pEnd = '\0';
    return;
}

void
pmuStrTrim
(
    char **ppStr
)
{
    char *pTmp;
    if ((ppStr == NULL) || (*ppStr == NULL))
        return;

    pTmp = *ppStr;
    while (*pTmp == ' ')
        pTmp++;
    *ppStr = pTmp;

    while (*pTmp != ' ')
        pTmp++;
    *pTmp = '\0';
}

/*!
 * Return symbol file path from directory of LWW_MANUAL_SDK. 
 *
 * @return Symbol file path
 */
const char*
pmuGetSymFilePath()
{
    return DIR_SLASH "pmu" DIR_SLASH "bin";
}

/*!
 * Return string of engine name
 *
 * @return Engine Name
 */
const char*
pmuGetEngineName()
{
    return "PMU";
}

/*!
 * Init and return the Falcon object that presents the PMU
 *
 * @return the Falcon object of the PMU
 */
POBJFLCN                
pmuGetFalconObject()
{
    // Init the object if it is not done yet
    if (!pmuFlcn.engineName)
    {
        pmuFlcn.pFCIF = pPmu[indexGpu].pmuGetFalconCoreIFace();
        pmuFlcn.pFEIF = pPmu[indexGpu].pmuGetFalconEngineIFace();

        if(pmuFlcn.pFEIF)
        {
            pmuFlcn.engineName = pmuFlcn.pFEIF->flcnEngGetEngineName();
            pmuFlcn.engineBase = pmuFlcn.pFEIF->flcnEngGetFalconBase();
        }
        else
        {    
            pmuFlcn.engineName = "PMU";
        }
        sprintf(pmuFlcn.symPath, "%s%s", LWSYM_VIRUTAL_PATH, "pmusym/");
        pmuFlcn.bSympathSet = TRUE;
    }

    return &pmuFlcn;
}

/*!
 * 
 * @return LwU32 DMEM access port for LwWatch
 */
LwU32
pmuGetDmemAccessPort()
{
    return 2;
}

