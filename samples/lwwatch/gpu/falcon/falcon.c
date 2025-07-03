
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  falcon.c
 * @brief Common Falcon Non-Hal functions.
 */

/* ------------------------ Includes --------------------------------------- */
#include "print.h"
#include "falcon.h"


// RM header for command/message interfaces to communicate with PMU
#include "rmpmucmdif.h"

// RM header for command/message interfaces to communicate with DPU
#include "rmdpucmdif.h"

// RM header for command/message interfaces to communicate with SEC2
#include "rmsec2cmdif.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */
static const char * pmuGetTaskName(LwU32);
static const char * dpuGetTaskName(LwU32);
static const char * sec2GetTaskName(LwU32);
/* ------------------------ Defines ---------------------------------------- */

void flcnRegWr32(LwU32 addr, LwU32 value, LwU32 engineBase)
{   
    extern  POBJFLCN   thisFlcn;
    if(thisFlcn && IsTegra())
    {
        DEV_REG_WR32(addr + engineBase, value, thisFlcn->engineName, 0);
    }
    else
    {
        GPU_REG_WR32(addr + engineBase, value);
    }
}

LwU32 flcnRegRd32(LwU32 addr, LwU32 engineBase)
{
    extern  POBJFLCN   thisFlcn;
    if(thisFlcn && IsTegra())
    {
        return DEV_REG_RD32(addr + engineBase, thisFlcn->engineName, 0);
    }
    else
    {
        return GPU_REG_RD32(addr + engineBase );
    }
}

/*!
 * Dump the contents of a Falcon queue. Parsing is not lwrrently implemented.
 *
 * @param bParse[in]      Try to parse the queue?
 * @param pQueue[in]      Pointer to queue structure
 * @param engineName[in]  String of engine name
 */
void
flcnQueueDump
(
    LwBool          bParse,
    PFLCN_QUEUE     pQueue,
    const char*     engineName
)
{
    LwU32 sizeInBytes;

    if (bParse)
    {
        dprintf("lw: %s: Parsing is not supported yet\n", __FUNCTION__);
        return;
    }

    sizeInBytes = pQueue->length * sizeof(LwU32);
    dprintf("lw:\t%s QUEUE 0x%02x: tail=0x%04x, head=0x%04x",
            engineName, pQueue->id, pQueue->tail, pQueue->head);

    if (sizeInBytes == 0)
    {
        dprintf(" - empty\n");
    }
    else
    {
        dprintf("\n");
        printBuffer((char*)pQueue->data, sizeInBytes, pQueue->tail, 04);
    }
}

void
flcnDmemDump
(
    const FLCN_ENGINE_IFACES *pFEIF,
    LwU32 offset,
    LwU32 lengthInBytes,
    LwU8  port,
    LwU8  size
)
{
    const FLCN_CORE_IFACES *pFCIF = pFEIF->flcnEngGetCoreIFace();
    const char  *engineName = pFEIF->flcnEngGetEngineName();
    LwU32        engineBase  = pFEIF->flcnEngGetFalconBase();
    LwU32    memSize         = 0x0;
    LwU32    numPorts        = 0x0;
    LwU32    length          = 0x0;
    LwU32*   buffer          = NULL;
    LwU32    dmemVaBound     = 0x0;
    LwBool   bIsAddrVa       = LW_FALSE;

    // Tidy up the length to be 4-byte aligned
    lengthInBytes = (lengthInBytes + 3) & ~3ULL;
    offset = offset & ~3ULL;

    if (pFCIF->flcnIsDmemAccessAllowed != NULL)
    {
        if (!pFCIF->flcnIsDmemAccessAllowed(pFEIF,
                                            engineBase,
                                            offset,
                                            (offset + lengthInBytes),
                                            LW_TRUE)) // DMEM read
        {
            dprintf("lw: %s: DMEM access not permitted\n",
                    __FUNCTION__);
            return;
        }
    }

    // Get the size of the DMEM and number of DMEM ports
    memSize  = pFCIF->flcnDmemGetSize(engineBase);
    numPorts = pFCIF->flcnDmemGetNumPorts(engineBase);

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
                engineName,
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
 * Wrapper for DMEM read that performs security-related error checks before
 * allowing write.
 *
 * @param[in]  pFEIF      Pointer to the Falcon Engine Interface
 * @param[in]  offset     Offset in DMEM to write to
 * @param[in]  value      Value to write to DMEM
 * @param[in]  width      Width of value (1=byte, 2=half-word, 3=word)
 * @param[in]  length     Length (number of entries of width w)
 * @param[in]  port       DMEM port to use to write
 */
void
flcnDmemWrWrapper
(
    const FLCN_ENGINE_IFACES *pFEIF,
    LwU32 offset,
    LwU32 value,
    LwU32 width,
    LwU32 length,
    LwU8  port
)
{
    const FLCN_CORE_IFACES *pFCIF = pFEIF->flcnEngGetCoreIFace();
    LwU32        engineBase = pFEIF->flcnEngGetFalconBase();

    // Tidy up the offset to be 4-byte aligned
    offset = offset & ~3ULL;

    if (pFCIF->flcnIsDmemAccessAllowed != NULL)
    {
        if (!pFCIF->flcnIsDmemAccessAllowed(pFEIF,
                                            engineBase,
                                            offset,
                                            (offset + length),
                                            LW_FALSE)) // DMEM write
        {
            dprintf("lw: %s: DMEM access not permitted\n",
                    __FUNCTION__);
            return;
        }
    }

    // Actually write to the DMEM
    pFCIF->flcnDmemWrite(engineBase,
                         (LwU32)offset,
                         LW_FALSE,
                         (LwU32)value,
                         width,         // Width of each entry (in bytes)
                         length,        // Number of entries to write
                         (LwU32)port);
}

/*!
 * For each code tag between minTag and maxTag, check to see if its
 * loaded/mapped into IMEM for thisFlcn object. thisFlcn should be
 * probably set and restored if necessary by the caller.
 * Dump the reults in an easy to read table.
 *
 * @param[in]  minTag  Code tag to start the dump with
 * @param[in]  maxTag  Code tag to end the dump with
 * @param[in]  bSkipUnmappedTags
 *    TRUE if nothing should be printed for any tags that are not mapped
 */
void
flcnImemMapDump
(
    LwU32   minTag,
    LwU32   maxTag,
    BOOL    bSkipUnmapped
)
{
    const   FLCN_CORE_IFACES    *pFCIF  = NULL;
    extern  POBJFLCN thisFlcn;
    LwU32       tag             = 0x0;
    FLCN_TAG    tagInfo;
    BOOL        bIncludeSymbols = FALSE;
    FLCN_SYM    *pMatches       = NULL;
    FLCN_SYM    *pSym           = NULL;
    LwU32       engineBase      = 0x0;

    // Sanity checking
    if (!thisFlcn || !thisFlcn->pFEIF || !thisFlcn->pFCIF)
    {
        return;
    }

    pFCIF = thisFlcn->pFCIF;
    engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();
    bIncludeSymbols = thisFlcn->bSymLoaded;

    // Print out a pretty little header
    dprintf("lw:\tDumping IMEM tag to block mapping\n");
    dprintf("lw:\t----------------------------------------------------------------------------------\n");

    // Now we know how many tags to look through
    for (tag = minTag; tag <= maxTag; tag++)
    {
        pFCIF->flcnImemTag(engineBase, (tag << 8), &tagInfo);
        switch (tagInfo.mapType)
        {
            case FALCON_TAG_UNMAPPED:
                if (!bSkipUnmapped)
                {
                    dprintf("lw:\tTag 0x%02x: Not mapped to a block\n", tag);
                }
                break;

            case FALCON_TAG_MULTI_MAPPED:
            case FALCON_TAG_MAPPED:
                dprintf("lw:\tTag 0x%02x: block=0x%02x, valid=%d, pending=%d, secure=%-1d",
                        tag, tagInfo.blockInfo.blockIndex, tagInfo.blockInfo.bValid,
                        tagInfo.blockInfo.bPending, tagInfo.blockInfo.bSelwre);

                if (bIncludeSymbols)
                {
                    pMatches = flcnSymResolve(tag << 8);
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

                if (tagInfo.mapType == FALCON_TAG_MULTI_MAPPED)
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
                "to load the symbols for this falcon using '!flcn -<engine> load' first\n");
        dprintf("lw:\trecommended using !dpusym for loading dpu symbols in mgpu system\n");
    }
}

/*!
 * Simple bootstrap function for the DPU
 *
 * @param[in]  pFEIF      Pointer to the Falcon Engine Interface
 * @param[in]  pFilename  Pointer to binary file containing Falcon app
 */
void
flcnSimpleBootstrap
(
    const FLCN_ENGINE_IFACES *pFEIF,
    const char *pFilename
)
{
    const   FLCN_CORE_IFACES    *pFCIF  = pFEIF->flcnEngGetCoreIFace();

    LwU32   csize, dsize, totalsize;
    LwU32   i, blockCount, wordCount, byteCount;
    size_t  nread = 0;
    FILE   *pFile;
    char    line[1024];
    LwU8   *pBin = NULL;
    LwU32   engineBase = 0x0;
    LwU32   bytesWritten;
    LwBool  bWriteSuccess = LW_TRUE;

    // Get engine base
    engineBase = pFEIF->flcnEngGetFalconBase();

    pFile = fopen(pFilename, "rb");
    if (pFile == NULL)
    {
        dprintf("lw: Error: cannot open Falcon program: %s\n", pFilename);
        return;
    }

    // Read DPU binary signature.
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open Falcon program: %s\n", pFilename);
        return;
    }
    nread = strlen(line);
    line[nread - 1] = '\0';

    //Check Falcon identifier string at the start of the binary
    if(!strcmp(pFEIF->flcnEngGetEngineName(), "DPU"))
    {
        //DPU_ if we are trying to bootstrap DPU
        if (strcmp(line, "DPU_"))
        {
            dprintf("lw: Error: Invalid DPU Simple Program: %s\n", pFilename);
            return;
        }
    }else
    {
        if(!strcmp(pFEIF->flcnEngGetEngineName(), "PMU"))
        {
            //PMU_ if we are trying to bootstrap PMU
            if (strcmp(line, "PMU_"))
            {
                dprintf("lw: Error: Invalid PMU Simple Program: %s\n", pFilename);
                return;
            }
        }
        else
        {
                dprintf("lw: Error: Engine lwrrently not supported: %s\n", pFilename);
                return;
        }
    }

    // Read Code size
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open Falcon program: %s\n", pFilename);
        return;
    }
    if (sscanf(line, "%d\n", &csize) != 1)
    {
        dprintf("lw: Error: Invalid Falcon Simple Program: %s\n", pFilename);
        return;
    }

    // Read Data size
    if (!fgets(line, sizeof(line), pFile))
    {
        dprintf("lw: Error: cannot open Falcon program: %s\n", pFilename);
        return;
    }
    if (sscanf(line, "%d\n", &dsize) != 1)
    {
        dprintf("lw: Error: Invalid Falcon Simple Program: %s\n", pFilename);
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
            pFCIF->flcnImemSetTag(engineBase, (i * 4) >> 8, 0);
        }

        pFCIF->flcnImemWrite(engineBase, i*4, ((LwU32 *)pBin)[i], 4, 1, 0);

        dprintf("lw: Loading ucode....[%c]\r", "\\|/-"[i%4]);
        fflush(stdout);
    }
    // copy the rest
    byteCount = csize % 4;
    if (byteCount)
    {
        if ((wordCount % (256/4)) == 0)
        {
            pFCIF->flcnImemSetTag(engineBase, (wordCount * 4) >> 8, 0);

        }
        for (i = 0 ; i < byteCount ; i++)
        {
            pFCIF->flcnImemWrite(engineBase, wordCount * 4 + i, pBin[(wordCount*4) + i], 1, 1, 0);
        }
    }

    // write dummy data at the end of IMEM block to mark the last block 'valid'
    if (csize % 256)
    {
        blockCount = csize/256 + 1;
        //                           addr, data, width, size, port
        pFCIF->flcnImemWrite(engineBase, blockCount * 256 - 4, 0, 4, 1, 0);
    }


    // 2. copy data to DMEM
    // copy by words first
    wordCount = dsize / 4;
    for (i = 0; i < wordCount; i++)
    {
        bytesWritten = pFCIF->flcnDmemWrite(engineBase, i*4, LW_FALSE,
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
            bytesWritten = pFCIF->flcnDmemWrite(engineBase, wordCount * 4, LW_FALSE,
                                        (pBin + csize)[(wordCount*4) + i], 1, 1, 0);

            if (bytesWritten != 1)
            {
                dprintf("lw: Failed to write at offset 0x%x\n", wordCount * 4);
                bWriteSuccess = LW_FALSE;
                break;
            }
        }
        // free memory
    }
    free(pBin);
    fclose(pFile);

    if (bWriteSuccess)
    {
        // start at 0
        // Bootstrap Function
        pFCIF->flcnBootstrap(engineBase,0);

        dprintf("lw:\nlw: Starting %s!\n", pFilename);
    }
    else
    {
        dprintf("lw:\nlw: Cannot start %s!\n", pFilename);
    }
}

LwBool
flcnIsDmemAccessAllowed_STUB
(
    const FLCN_ENGINE_IFACES   *pFEIF,
    LwU32                       engineBase,
    LwU32                       addrLo,
    LwU32                       addrHi,
    LwBool                      bIsRead
)
{
    return LW_TRUE;
}

BOOL
flcnDmemBlk_STUB
(
    LwU32           engineBase,
    LwU32           blockIndex,
    FLCN_BLOCK      *pBlockInfo
)
{
    return LW_FALSE;
}

LwU32
flcnDmemGetTagWidth_STUB
(
    LwU32 engineBase
)
{
    return 0;
}

BOOL
flcnDmemTag_STUB
(
    LwU32           engineBase,
    LwU32           codeAddr,
    FLCN_TAG*       pTagInfo
)
{
    return LW_FALSE;
}

LwU32
flcnDmemVaBoundaryGet_STUB
(
    LwU32 engineBase
)
{
    return LW_FLCN_DMEM_VA_BOUND_NONE;
}

/*
 * Retrieve task name given the task ID for PMU
 *
 * @param[in]  taskID   id of the task
 *
 * @return     Task name
 */
static const char *
pmuGetTaskName
(
    LwU32 taskID
)
{
    // Make sure pmuGetTaskName is updated when we add new tasks.
#if RM_PMU_TASK_ID__END != 0x17
    #error "Please update pmuGetTaskName with the newly added tasks"
#endif

    switch (taskID)
    {
        case RM_PMU_TASK_ID__IDLE:
            return "IDLE";
        case RM_PMU_TASK_ID_CMDMGMT:
            return "CMDMGMT";
        case RM_PMU_TASK_ID_GCX:
            return "GCX";
        case RM_PMU_TASK_ID_LPWR:
            return "LPWR";
        case RM_PMU_TASK_ID_LPWR_LP:
            return "LPWR_LP";
        case RM_PMU_TASK_ID_WATCHDOG:
            return "WATCHDOG";
        case RM_PMU_TASK_ID_I2C:
            return "I2C";
        case RM_PMU_TASK_ID_SEQ:
            return "SEQ";
        case RM_PMU_TASK_ID_PCM:
            return "PCM";
        case RM_PMU_TASK_ID_PCMEVT:
            return "PCMEVT";
        case RM_PMU_TASK_ID_PMGR:
            return "PMGR";
        case RM_PMU_TASK_ID_PERFMON:
            return "PERFMON";
        case RM_PMU_TASK_ID_DISP:
            return "DISP";
        case RM_PMU_TASK_ID_THERM:
            return "THERM";
        case RM_PMU_TASK_ID_HDCP:
            return "HDCP";
        case RM_PMU_TASK_ID_ACR:
            return "ACR";
        case RM_PMU_TASK_ID_SPI:
            return "SPI";
        case RM_PMU_TASK_ID_PERF:
            return "PERF";
        case RM_PMU_TASK_ID_LOWLATENCY:
            return "LOWLATENCY";
        case RM_PMU_TASK_ID_PERF_DAEMON:
            return "PERF_DAEMON";
        case RM_PMU_TASK_ID_BIF:
            return "BIF";
        case RM_PMU_TASK_ID_PERF_CF:
            return "PERF_CF";
        case RM_PMU_TASK_ID_NNE:
            return "NNE";
        default:
            return "!error!";
    }
}

/*
 * Retrieve task name given the task ID for DPU
 *
 * @param[in]  taskID   id of the task
 *
 * @return     Task name
 */
static const char *
dpuGetTaskName
(
    LwU32 taskID
)
{
    // Make sure dpuGetTaskName is updated when we add new tasks.
#if RM_DPU_TASK_ID__END != 0xA
    #error "Please update dpuGetTaskName with the newly added tasks"
#endif

    switch (taskID)
    {
        case RM_DPU_TASK_ID__IDLE:
            return "IDLE";
        case RM_DPU_TASK_ID_DISPATCH:
            return "DISPATCH";
        case RM_DPU_TASK_ID_REGCACHE:
            return "REGCACHE";
        case RM_DPU_TASK_ID_VRR:
            return "VRR";
        case RM_DPU_TASK_ID_HDCP:
            return "HDCP";
        case RM_DPU_TASK_ID_HDCP22WIRED:
            return "HDCP22WIRED";
        case RM_DPU_TASK_ID_SCANOUTLOGGING:
            return "SCANOUTLOGGING";
        case RM_DPU_TASK_ID_MSCGWITHFRL:
            return "MSCGWITHFRL";
        case RM_DPU_TASK_ID_WKRTHD:
            return "WKRTHD";
        default:
            return "!error!";
    }
}

/*
 * Retrieve task name given the task ID for SEC2
 *
 * @param[in]  taskID   id of the task
 *
 * @return     Task name
 */
static const char *
sec2GetTaskName
(
    LwU32 taskID
)
{
    // Make sure sec2GetTaskName is updated when we add new tasks.
#if RM_SEC2_TASK_ID__END != 0x10
    #error "Please update sec2GetTaskName with the newly added tasks"
#endif

    switch (taskID)
    {
        case RM_SEC2_TASK_ID__IDLE:
            return "IDLE";
        case RM_SEC2_TASK_ID_CMDMGMT:
            return "CMDMGMT";
        case RM_SEC2_TASK_ID_CHNMGMT:
            return "CHNMGMT";
        case RM_SEC2_TASK_ID_RMMSG:
            return "RMMSG";
        case RM_SEC2_TASK_ID_WKRTHD:
            return "WKRTHD";
        case RM_SEC2_TASK_ID_HDCPMC:
            return "HDCPMC";
        case RM_SEC2_TASK_ID_GFE:
            return "GFE";
        case RM_SEC2_TASK_ID_HWV:
            return "HWV";
        case RM_SEC2_TASK_ID_LWSR:
            return "LWSR";
        case RM_SEC2_TASK_ID_PR:
            return "PLAYREADY";
        case RM_SEC2_TASK_ID_VPR:
            return "VPR";
        case RM_SEC2_TASK_ID_HDCP22WIRED:
            return "HDCP22WIRED";
        case RM_SEC2_TASK_ID_ACR:
            return "ACR";
        case RM_SEC2_TASK_ID_HDCP1X:
            return "HDCP1X";
        case RM_SEC2_TASK_ID_APM:
            return "APM";
        case RM_SEC2_TASK_ID_SPDM:
            return "SPDM";
        case RM_SEC2_TASK_ID_WORKLAUNCH:
            return "WORKLAUNCH";
        default:
            return "!error!";
    }
}

/*
 * Retrieve task name given the task ID
 *
 * @param[in]  taskID   id of the task
 *
 * @return     Task name
 */
const char *
flcnGetTasknameFromId_STUB
(
    LwU32 taskId
)
{
    extern  POBJFLCN thisFlcn;
    const FLCN_ENGINE_IFACES *pFEIF = thisFlcn->pFEIF;

    if (pFEIF == NULL)
    {
        return NULL;
    }

    if(!strcmp(pFEIF->flcnEngGetEngineName(), "DPU"))
    {
        return dpuGetTaskName(taskId);
    }
    else if(!strcmp(pFEIF->flcnEngGetEngineName(), "PMU"))
    {
        return pmuGetTaskName(taskId);
    }
    else if(!strcmp(pFEIF->flcnEngGetEngineName(), "SEC2"))
    {
        return sec2GetTaskName(taskId);
    }

    return NULL;
}

void flcnTrpcClear
(
    LwBool  bClearObjdump,
    LwBool  bClearExtPC
)
{
    extern  POBJFLCN thisFlcn;
    LwU32 index = 0;
    if (thisFlcn == NULL)
    {
        return;
    }

    if (bClearObjdump)
    {
        thisFlcn -> bObjdumpFileLoaded = LW_FALSE;
        thisFlcn -> objdumpFileSize = 0;
        thisFlcn -> objdumpFileFuncN = 0;
        if (thisFlcn -> pObjdumpBuffer != NULL)
        {
            free(thisFlcn -> pObjdumpBuffer);
            thisFlcn -> pObjdumpBuffer = NULL;
        }
        if (thisFlcn -> ppObjdumpFileFunc != NULL)
        {
            for (index = 0; index < thisFlcn -> objdumpFileFuncN; ++ index)
            {
                if (thisFlcn -> ppObjdumpFileFunc[index] != NULL)
                {
                    free(thisFlcn -> ppObjdumpFileFunc[index]);
                    thisFlcn -> ppObjdumpFileFunc[index] = NULL;
                }
            }
            free(thisFlcn -> ppObjdumpFileFunc);
            thisFlcn -> ppObjdumpFileFunc = NULL;
        }
    }

    if (bClearExtPC)
    {
        if (thisFlcn -> pExtTracepcBuffer != NULL)
        {
            for (index = 0; index < thisFlcn -> extTracepcNum; ++ index)
            {
                if (thisFlcn -> pExtTracepcBuffer[index] != NULL)
                {
                    free(thisFlcn -> pExtTracepcBuffer[index]);
                }
                thisFlcn -> pExtTracepcBuffer[index] = NULL;
            }
            free(thisFlcn -> pExtTracepcBuffer);
            thisFlcn -> pExtTracepcBuffer = NULL;
        }
        thisFlcn -> extTracepcNum = 0;
    }
}

