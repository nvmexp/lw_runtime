/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmuswak.c
 * @brief WinDbg Extension for PMU.
 */

/* ------------------------ Includes --------------------------------------- */
#include "pmu.h"
#include "lwsym.h"
#include "print.h"
#include "chip.h"
#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
  #include <unistd.h>
#endif

/* ------------------------ Types definitions ------------------------------ */
typedef struct _OverlayInfo OverlayInfo;

/** @struct _OverlayInfo
 *
 *  structure used to store information about overlays
 */
struct _OverlayInfo
{
    char  overlayName[PMU_SYM_NAME_SIZE];
    LwU32 addrStart;
    LwU32 addrEnd;
};

/** @enum ADDRTYPE
 *
 *  Used to represent different type of memory, for statistics use
 */
typedef enum
{
    ADDRTYPE_TCB,
    ADDRTYPE_PVTTCB,
    ADDRTYPE_QUEUE,
    ADDRTYPE_QUEUESTORAGE,
    ADDRTYPE_TASKSTACK,
    ADDRTYPE_RESIDENT
} ADDRTYPE;

/** @struct AddrInfo
 *
 *  Used to collection information of each memory block
 */
typedef struct
{
    ADDRTYPE type;
    LwU32    startAddr;
    char     msg[256];
    LwU32    endAddr;
} AddrInfo, *PAddrInfo;

/** @struct Linkedlist
 *
 *  structure to store information of each memory block
 */
typedef struct LinkedList
{
    AddrInfo            addrInfo;
    struct LinkedList  *pNext;
} LinkedList, *PLinkedList;

/* ------------------------ Function Prototypes ---------------------------- */
static  const char     *pmuGetAddrTypeName   (ADDRTYPE);
static  LwBool          pmuHalted            (void);
static  void            dumpMemStructure     (PLinkedList);
static  void            dumpSymbolContents   (LwU8 *, LwU32, PLinkedList *);
static  void            dumpStackTrace       (PMU_TCB *, LwBool, LwU32);
static  void            dumpTcbAndStacktrace (PLinkedList *);
static  LwBool          dumpImemInfo         (void);
static  LwBool          loadDmem             (LwU8 **, LwU32 *,LwU32);
static  LwBool          loadImemTagBlock     (PmuTagBlock **, LwU32 *);
static  void            dumpDmemInfo         (LwU8 *, LwU32, PLinkedList);
static  void            AddInfoToList        (PLinkedList *, PLinkedList);
static  void            parseExciReg         (void);
static  void            dumpGPUInfo          (void);
static  void            dumpPMURunStatus     (void);

/* ------------------------ Globals ---------------------------------------- */
#if defined(WIN32)
extern  void            openLogFile          (const char *);
extern  void            closeLogFile         (void);
#endif

/* ------------------------ Defines ---------------------------------------- */

#define PMU_PRINT_SEPARATOR() dprintf("lw:========================================\n")

static void
dumpGPUInfo(void)
{
    char *pChipManualsDir[MAX_PATHS];
    char *pClassNum;
    int numPaths = 1;
    int i = 0;

    for(i = 0; i < MAX_PATHS; i++)
    {
        pChipManualsDir[i] = (char *)malloc(32  * sizeof(char));
    }
    pClassNum = (char *)malloc(32 * sizeof(char));

    if(!GetManualsDir(pChipManualsDir, pClassNum, &numPaths))
    {
        dprintf("\n: Unknown or unsupported lWpu GPU. Ensure that %s() supports the chip you're working on.\n",
            __FUNCTION__);
        goto Cleanup;
    }

    dprintf("   GPU Name:   %s\n", pChipManualsDir[0]);

Cleanup:
    // Lets free the char array
    for(i = 0; i < MAX_PATHS; i++)
    {
        free(pChipManualsDir[i]);
    }
    free(pClassNum);
}

/*!
 * Map ADDRTYPE to corresponding string for output usage
 *
 * @param[in]  addrType      Enum represents the Address Type
 *
 * @return  constant string
 *      On Success
 * @return  NULL
 *      In Failure
 */
static const char *
pmuGetAddrTypeName
(
    ADDRTYPE addrType
)
{
    switch (addrType)
    {
    case ADDRTYPE_TCB:
        return "Tcb Struct";
    case ADDRTYPE_PVTTCB:
        return "Private Tcb";
    case ADDRTYPE_QUEUE:
        return "Queue";
    case ADDRTYPE_QUEUESTORAGE:
        return "Queue Store";
    case ADDRTYPE_TASKSTACK:
        return "Task Stack";
    default:
        return NULL;
    }
}

/*!
 * This function continuously read the value of register PC to
 * determine if PMU is halted or running.
 *
 * @return LW_TRUE
 *     PMU is halted
 * @return LW_FALSE
 *     PMU is running
 */
static LwBool
pmuHalted(void)
{
    LwU32 prevPC;
    LwU32 lwrPC;
    LwU32 checks = 100;
    LwU32 i;
    LwU32 interval = 30;

    prevPC = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_PC);
    for (i = 0; i < checks; ++i)
    {
#ifdef WIN32
        Sleep(interval);
#elif defined CLIENT_SIDE_RESMAN
        // Don't sleep in MODS with RM
#else
        sleep(interval);
#endif
        lwrPC = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_PC);
        if (lwrPC != prevPC)
        {
            return LW_FALSE;
        }
    }
    return LW_TRUE;
}

/*!
 * Print out memory structure
 *
 * @param[in]  PLinkedList      A linked list stores every item in memory
 *
 */
static void
dumpMemStructure
(
    PLinkedList pHead
)
{
    PAddrInfo   pInfo;

    dprintf("lw:============================================\n");
    dprintf("lw:  Memory structure(those recognized by swak)\n");
    dprintf("lw:============================================\n");

    dprintf("lw:       Type                   Item          addrStart      addrEnd\n");
    while (pHead != NULL)
    {
        pInfo = &pHead->addrInfo;

        dprintf("lw: %12s     %20s       0x%05x       0x%05x\n", pmuGetAddrTypeName(pInfo->type), pInfo->msg, pInfo->startAddr, pInfo->endAddr);

        pHead = pHead->pNext;
    }
}

/*!
 * Read the symbol file from lwsym, write it to output
 *
 * @param[in]  PLinkedList      A linked list stores every item in memory
 *
 */
static void
dumpSymbolFiles
(
    char *pFileExtension
)
{
    char *pLwsymFilename;
    const char *pUcodeName = pPmu[indexGpu].pmuUcodeName();
    LwU8 *pFileBuffer      = NULL;
    LwU32 fileBufferSize;
    LwU32 status;
    LwU32 idx;

    pLwsymFilename = malloc(strlen(LWSYM_VIRUTAL_PATH) +
                            strlen("pmusym/")          +
                            strlen(pUcodeName)         +
                            strlen(pFileExtension)     +
                            (size_t)1);

    sprintf(pLwsymFilename, "%s%s%s%s",
                             LWSYM_VIRUTAL_PATH,
                             "pmusym/",
                              pUcodeName,
                              pFileExtension);

    status = lwsymFileLoad(pLwsymFilename, &pFileBuffer, &fileBufferSize);
    if (status == LWSYM_OK)
    {
        dprintf("lw:=====================================\n");
        dprintf("lw:   Dumping content of file %s\n", pLwsymFilename);
        dprintf("lw:=====================================\n");

        for (idx = 0; idx < fileBufferSize; ++idx)
        {
            if (pFileBuffer[idx] != '\r')
                dprintf("%c", pFileBuffer[idx]);
        }
    }

    free(pLwsymFilename);
    free(pFileBuffer);
}

/*!
 * Print out every symbol in .nm file
 * If the symbol is of data type and its size is known, print out all the
 * the contents belongs to the variable the symbol represents.
 *
 * @param[in]  pDmem      Buffer that stores content of data memory
 * @param[in]  dmemSize   Size of the buffer
 *
 */
static void
dumpSymbolContents
(
    LwU8         *pDmem,
    LwU32         dmemSize,
    PLinkedList  *ppHead
)
{
    PMU_SYM    *pMatches;
    BOOL        bExactFound;
    LwU32       count;
    LwU32       i                   = 0;
    LwU32       rsdend              = 0;
    LwU32       dataParsedInRsd     = 0;
    const char *pSymResidentDataEnd = "_resident_data_end";
    //
    // Record how many we know about the resident aread
    // 0x0000 ~ _resident_data_end
    //
    pMatches = pmuSymFind(pSymResidentDataEnd, FALSE, &bExactFound, &count);
    if (!bExactFound)
    {
        dprintf("ERROR: Unable to find symbol %s, please check LwWatch!\n", pSymResidentDataEnd);
    }
    else
    {
        rsdend = pMatches->addr;
    }

    pMatches = pmuSymFind("_", FALSE, &bExactFound, &count);

    dprintf("lw:=====================================\n");
    dprintf("nlw:   Dumping symbols with their values\n");
    dprintf("lw:=====================================\n");

    while (pMatches != NULL)
    {
        pmuSymPrintBrief(pMatches, i++);

        if ((!pMatches->bSizeEstimated) &&
            (pMatches->section != 'T') &&
            (pMatches->section != 't') &&
            (pMatches->addr < dmemSize) &&
            (pMatches->addr + pMatches->size < dmemSize))
        {
            LwU32 alignedSize =(pMatches->size + 3) >> 2 << 2;

            printBuffer((char*)&pDmem[pMatches->addr], alignedSize, pMatches->addr, 0x4);
            dprintf("\n");

            if (pMatches->addr < rsdend)
            {
                // Data should not pass resident data end
                if (pMatches->addr + pMatches->size > rsdend)
                {
                    dprintf("ERROR:   The above data is not valid, it should not pass resident data end 0x%05x!!!", rsdend);
                }
                dataParsedInRsd += pMatches->size;
            }
        }
        pMatches = pMatches->pTemp;
    }

    // Put resident parsed statistics into linked list
    {
        PLinkedList pItem = (PLinkedList)malloc(sizeof(LinkedList));
        PAddrInfo   pInfo = &pItem->addrInfo;

        pInfo->startAddr = dataParsedInRsd;
        pInfo->endAddr = rsdend;
        pInfo->type = ADDRTYPE_RESIDENT;
        pInfo->msg[0] = '\0';

        AddInfoToList(ppHead, pItem);
    }
}

/*!
 * Added an item to a linked list in a ascending order (based on addrInfo.startaddr)
 *
 * @param[in][out]  ppHead      Header of the linked list, may be changed in the function
 * @param[in]       pItem       Item to be added
 *
 */
void
AddInfoToList
(
    PLinkedList *ppHead,
    PLinkedList  pItem
)
{
    pItem->pNext = NULL;
    if (*ppHead == NULL)
    {
        *ppHead = pItem;
    }
    else
    {
        PLinkedList pTemp = *ppHead;
        PLinkedList pPrev = NULL;
        PAddrInfo   pLwrInfo = &pTemp->addrInfo;
        PAddrInfo   pAddedInfo = &pItem->addrInfo;

        while (pAddedInfo->startAddr > pLwrInfo->startAddr)
        {
            pPrev = pTemp;
            pTemp = pTemp->pNext;
            if (pTemp == NULL)
                break;

            pLwrInfo = &pTemp->addrInfo;
        }

        if (pTemp == NULL)
        {
            pPrev->pNext = pItem;
            return;
        }
        else if (pPrev == NULL)
        {
            pItem->pNext = *ppHead;
            *ppHead = pItem;
            return;
        }
        else
        {
            pPrev->pNext = pItem;
            pItem->pNext = pTemp;
            return;
        }
    }
}

/*!
 * Dump stack trace for a task
 *
 * @param[in]  pTcb       The task being dump
 * @param[in]  bLwrrent   Whether the task is the current one or not
 * @param[in]  port       Port used to read data
 *
 */
static void
dumpStackTrace
(
    PMU_TCB *pTcb,
    LwBool   bLwrrent,
    LwU32    port
)
{
    PMU_TCB_PVT *pPrivTcb;
    LwU32        ucodeVersion;
    LwU32        pmuTcbPStack = 0;
    LwU32        pmuTcbPStackSize = 0;

    ucodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();

    pmustLoad(NULL, ucodeVersion, FALSE);
    pmuTcbGetPriv(&pPrivTcb, (LwU32)pTcb->pmuTcb.pmuTcb1.pvTcbPvt, port);

    // grab pmu pvt tcb specific variables
    switch (pPrivTcb->tcbPvtVer)
    {
        case PMU_TCB_PVT_VER_0:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.stackSize;
            break;

        case PMU_TCB_PVT_VER_1:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.stackSize;
            break;

        case PMU_TCB_PVT_VER_2:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.stackSize;
            break;

        case PMU_TCB_PVT_VER_3:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.stackSize;
            break;

        case PMU_TCB_PVT_VER_4:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.stackSize;
            break;

        case PMU_TCB_PVT_VER_5:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.stackSize;
            break;

        case PMU_TCB_PVT_VER_6:
            pmuTcbPStack    = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.pStack;
            pmuTcbPStackSize = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.stackSize;
            break;

        default:
            dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
            break;
    }

    if (pTcb->tcbVer == PMU_TCB_VER_4)
    {
        pmuTcbPStack = pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress;
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_3)
    {
        pmuTcbPStack = pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress;
    }

    if (bLwrrent)
    {
        // For current task, call pmust to dump the stack trace
        pmustPrintStacktrace(pmuTcbPStack, pmuTcbPStackSize, ucodeVersion);
    }
    else
    {
        LwU32 contentSize;
        LwU32 alignedVersion;
        LwU32 stackBottom = pmuTcbPStack + (pmuTcbPStackSize << 2);
        LwU8 *pStack;
        LwU32 pc;
        LwU32 index;
        LwU32 wordsRead;

        contentSize = stackBottom - pTcb->pmuTcb.pmuTcb1.pxTopOfStack;
        alignedVersion = (contentSize + sizeof(LwU32) - 1) / sizeof(LwU32);
        pStack = (LwU8 *)malloc(alignedVersion << 2);
        wordsRead = pPmu[indexGpu].pmuDmemRead(pTcb->pmuTcb.pmuTcb1.pxTopOfStack,
                                               LW_TRUE,
                                               alignedVersion,
                                               port,
                                               (LwU32 *)pStack);

        if (wordsRead != alignedVersion)
        {
            dprintf("Cannot read current stack at address 0x%x\n",
                    pTcb->pmuTcb.pmuTcb1.pxTopOfStack);
        }
        else
        {
            //a0 - a15, CSW, uxCriticalNesting  so 18 x 4
            index = 72;

            //pc stored right here
            pc = *((int*)&pStack[index]);
            index += 4;

            pmustStacktraceForTasks(&pStack[index], contentSize - index, pmuTcbPStackSize << 2, 0, pc, ucodeVersion);
        }
        free(pStack);
    }
    free(pPrivTcb);
}


/*!
 * Dump TCB and stacktrace for the task
 *
 * @param[in]  ppHead        Linked list head pointer, the linked list is used to
 *                           store start address and size of the tcb-related structures
 *
 */
static void
dumpTcbAndStacktrace
(
    PLinkedList *ppHead
)
{
    LwU32    i;
    PMU_TCB *pTcb;
    PMU_TCB  lwrTcb;
    LwU32    nTcb;
    LwU32    port = 1;
    char     buf[256];

    if (!pmuSymCheckAutoLoad())
    {
        return;
    }

    if (!pmuTcbGetLwrrent(&lwrTcb, port))
    {
        dprintf("ERROR: Unable to get current tcb, please update LwWatch!!!\n");
        return;
    }

    if (!pmuTcbFetchAll(1, &pTcb, &nTcb))
    {
        dprintf("ERROR: Failed to fetch tcbs\n");
        return;
    }

    for (i = 0; i < nTcb; ++i)
    {
        if (pTcb[i].tcbAddr == lwrTcb.tcbAddr)
            break;
    }

    if (i > nTcb - 1)
    {
        dprintf("ERROR: Current Tcb is not in pmuTcbFetchAll, please fix this function before using pmuswak!!!\n");
        return;
    }

    //
    // Exchange the position of lwrrentTcb to position 0
    // Since the current tcb should be printed as the first one.
    //
    pTcb[i] = pTcb[0];
    pTcb[0] = lwrTcb;

    for (i = 0; i < nTcb; ++i)
    {
        PMU_PRINT_SEPARATOR();
        // Validate the tcb, print out error msg if it is corrupted
        if (pmuTcbValidate(&pTcb[i]) != 0)
        {
            pmuGetTaskNameFromTcb(&pTcb[i], buf, sizeof(buf), port);
            dprintf("*******************************************************************\n");
            dprintf("lw:   !ERROR!  TASK %s\n", buf);
            dprintf("lw:   The tcb structure is corrupted, possibly due to stack overflow.\n");
            dprintf("lw:   Please check if stack size of the task is large enough.\n");
            dprintf("*******************************************************************\n");
        }

        // Even though the tcb is not valid, we'll still dump out the structure
        // so that people have more ideas which part gone wrong
        {

            PAddrInfo pInfo;
            PLinkedList pList;
            PMU_TCB_PVT *pPrivTcb;

            switch (pTcb[i].tcbVer)
            {
            case PMU_TCB_VER_0:
                // Collect statistics for task stack
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = pTcb[i].pmuTcb.pmuTcb0.pStack;
                pInfo->endAddr = pInfo->startAddr + (pTcb[i].pmuTcb.pmuTcb0.stackDepth << 2) - 1;
                pInfo->type = ADDRTYPE_TASKSTACK;
                sprintf(pInfo->msg, "task %d (%s)", pTcb[i].pmuTcb.pmuTcb0.tcbNumber, pTcb[i].pmuTcb.pmuTcb0.taskName);

                AddInfoToList(ppHead, pList);

                // Collect statistics for task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = pTcb[i].tcbAddr;
                pInfo->endAddr = pInfo->startAddr + sizeof(pTcb[i].pmuTcb.pmuTcb0) - 1;
                pInfo->type = ADDRTYPE_TCB;
                sprintf(pInfo->msg, "task %d (%s)", pTcb[i].pmuTcb.pmuTcb0.tcbNumber, pTcb[i].pmuTcb.pmuTcb0.taskName);
                AddInfoToList(ppHead, pList);

                break;
            case PMU_TCB_VER_1:
                // Collect statistics for task stack
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pmuTcbGetPriv(&pPrivTcb, (LwU32)pTcb[i].pmuTcb.pmuTcb1.pvTcbPvt, 1);

                // grab pmu pvt tcb specific variables
                switch (pPrivTcb->tcbPvtVer)
                {
                    case PMU_TCB_PVT_VER_0:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt0.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_1:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt1.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_2:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt2.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_3:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt3.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_4:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt4.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_5:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt5.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_6:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt6.stackSize << 2) - 1;
                        break;

                    default:
                        dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                        break;
                }

                pInfo->type = ADDRTYPE_TASKSTACK;
                sprintf(pInfo->msg, "task %d (%s)", pTcb[i].pmuTcb.pmuTcb1.ucTaskID, pmuGetTaskName(pTcb[i].pmuTcb.pmuTcb1.ucTaskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = pTcb[i].tcbAddr;
                pInfo->endAddr = pInfo->startAddr + sizeof(pTcb[i].pmuTcb.pmuTcb1) - 1;
                pInfo->type = ADDRTYPE_TCB;
                sprintf(pInfo->msg, "task %d (%s)", pTcb[i].pmuTcb.pmuTcb1.ucTaskID, pmuGetTaskName(pTcb[i].pmuTcb.pmuTcb1.ucTaskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for private task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = (LwU32)pTcb[i].pmuTcb.pmuTcb1.pvTcbPvt;
                pInfo->endAddr = pInfo->startAddr + sizeof(RM_RTOS_TCB_PVT) - 1;
                pInfo->type = ADDRTYPE_PVTTCB;
                sprintf(pInfo->msg, "task %d (%s)", pTcb[i].pmuTcb.pmuTcb1.ucTaskID, pmuGetTaskName(pTcb[i].pmuTcb.pmuTcb1.ucTaskID));
                AddInfoToList(ppHead, pList);

                break;
            case PMU_TCB_VER_2:
                // Collect statistics for task stack
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pmuTcbGetPriv(&pPrivTcb, (LwU32)pTcb[i].pmuTcb.pmuTcb2.pvTcbPvt, 1);

                // grab pmu pvt tcb specific variables
                switch (pPrivTcb->tcbPvtVer)
                {
                    case PMU_TCB_PVT_VER_0:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt0.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt0.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_1:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt1.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt1.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_2:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt2.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt2.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_3:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt3.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt3.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_4:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt4.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt4.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_5:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt5.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt5.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_6:
                        pInfo->startAddr = pPrivTcb->pmuTcbPvt.pmuTcbPvt6.pStack;
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt6.stackSize << 2) - 1;
                        break;


                    default:
                        dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                        break;
                }

                pInfo->type = ADDRTYPE_TASKSTACK;
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID, pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = pTcb[i].tcbAddr;
                pInfo->endAddr = pInfo->startAddr + sizeof(pTcb[i].pmuTcb.pmuTcb2) - 1;
                pInfo->type = ADDRTYPE_TCB;
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID, pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for private task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = (LwU32)pTcb[i].pmuTcb.pmuTcb2.pvTcbPvt;
                pInfo->endAddr = pInfo->startAddr + sizeof(RM_RTOS_TCB_PVT) - 1;
                pInfo->type = ADDRTYPE_PVTTCB;
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID,  pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt2.taskID));
                AddInfoToList(ppHead, pList);

                break;
            case PMU_TCB_VER_3:                
                // Collect statistics for private TCB variables
                pmuTcbGetPriv(&pPrivTcb, (LwU32)pTcb[i].pmuTcb.pmuTcb3.pvTcbPvt, 1);
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->startAddr = pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress;
                switch (pPrivTcb->tcbPvtVer)
                {
                    case PMU_TCB_PVT_VER_0:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt0.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_1:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt1.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_2:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt2.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_3:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt3.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_4:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt4.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_5:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt5.stackSize << 2) - 1;
                        break;

                    case PMU_TCB_PVT_VER_6:
                        pInfo->endAddr   = pInfo->startAddr + (pPrivTcb->pmuTcbPvt.pmuTcbPvt6.stackSize << 2) - 1;
                        break;

                    default:
                        dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                        break;
                }
                pInfo->type = ADDRTYPE_TASKSTACK;                
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID, pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->type = ADDRTYPE_TCB;
                pInfo->startAddr = pTcb[i].tcbAddr;
                pInfo->endAddr = pInfo->startAddr + sizeof(pTcb[i].pmuTcb.pmuTcb3) - 1;
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID, pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID));
                AddInfoToList(ppHead, pList);

                // Collect statistics for private task control block
                pList = (PLinkedList)malloc(sizeof(LinkedList));
                pInfo = &pList->addrInfo;
                pInfo->type = ADDRTYPE_PVTTCB;
                pInfo->startAddr = pTcb[i].pmuTcb.pmuTcb3.pvTcbPvt;
                pInfo->endAddr = pTcb[i].pmuTcb.pmuTcb3.pvTcbPvt + sizeof(RM_RTOS_TCB_PVT) - 1;
                sprintf(pInfo->msg, "task %d (%s)", pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID, pmuGetTaskName(pPrivTcb->pmuTcbPvt.pmuTcbPvt1.taskID));
                AddInfoToList(ppHead, pList);

                break;
            default:
                dprintf("ERROR: Unknown tcb version, please update LwWatch pmuswak!!!\n");
                break;
            }

            // Special handling for current tcb since stackTop is not the latest value
            if (pTcb[i].tcbAddr == lwrTcb.tcbAddr)
            {
                dprintf("lw:         Current TCB\n");
                dprintf("lw:  Value StackTop below is the old value when the\n");
                dprintf("lw:  current task is switched out last time\n");
                PMU_PRINT_SEPARATOR();
            }
            pmuTcbDump(&pTcb[i], FALSE, 1, 4);

            PMU_PRINT_SEPARATOR();
            if (pTcb[i].tcbAddr != lwrTcb.tcbAddr)
            {
                dumpStackTrace(&pTcb[i], LW_FALSE, port);
            }
            else
            {
                dumpStackTrace(&pTcb[i], LW_TRUE, port);
            }
        }
    }
    if (nTcb > 0)
        PMU_PRINT_SEPARATOR();

    free(pTcb);
    return;
}

/*!
 * Load full data memory from PMU
 * This function dynamically allocates a space to store all the PMU data memory.
 *
 * @param[out]  ppData        Stores the pointer points to dynamically allocated
 *                            memory which stores all PMU data memory
 * @param[out]  pSize         Size of the data memory
 * @param[in]   port          Port used to read data from PMU
 *
 * @return     LW_TRUE
 *       Succeeded to load data memory
 * @return     LW_FALSE
 *       Falied to load data memory
 */
static LwBool
loadDmem
(
    LwU8  **ppData,
    LwU32  *pSize,
    LwU32   port
)
{
    LwU8 *pData;

    *pSize = pPmu[indexGpu].pmuDmemGetSize();
    pData = (LwU8 *)malloc(*pSize);

    if (!pData)
    {
        dprintf("ERROR: Unable to allocate memory!\n");
        return LW_FALSE;
    }

    pPmu[indexGpu].pmuDmemRead(0,
                              LW_FALSE,
                              (*pSize) >> 2,
                              port,
                              (LwU32 *)pData);
    *ppData = pData;
    return LW_TRUE;
}

/*!
 * Dump data memory information
 *
 * @param[in]   pData         Content of whole data memory
 * @param[in]   dmemSize      Size of data memory
 * @param[in]   pTaskList     Linked-list storing start-address and size
 *                            of each known structure
 *
 */
static void
dumpDmemInfo
(
    LwU8       *pData,
    LwU32       dmemSize,
    PLinkedList pTaskList
)
{
    RM_RTOS_DEBUG_ENTRY_POINT  debugEntry;
    PMU_SYM                   *pMatches;
    LwU32                      count;
    BOOL                       bExactFound;
    const char                *osDebugName      = "OsDebugEntryPoint";
    LwF32                      isrStackUsedPct;
    LwU32                      isrStackFree;
    LwU32                      isrStackSize;
    LwU32                      idx;
    LwU8                       stackFilled      = 0xa5;
    LwU32                      taskStackSize    = 0;
    LwU32                      tcbSize          = 0;
    LwU32                      pvtTcbSize       = 0;
    LwU32                      queueSize        = 0;
    LwU32                      queueStorageSize = 0;
    LwU32                      rsdarea          = 0;
    LwU32                      parsedRsdArea    = 0;
    PAddrInfo                  pAddrInfo;
    PLinkedList                pList;

    // This is the place we get information about used heap
    pMatches = pmuSymFind(osDebugName, FALSE, &bExactFound, &count);

    if (!bExactFound)
    {
        dprintf("Unable to find symbol %s, please update LwWatch\n", osDebugName);
        return;
    }

    memcpy(&debugEntry, &pData[pMatches->addr], sizeof(RM_RTOS_DEBUG_ENTRY_POINT));
    isrStackSize = sizeof(LwU32) * debugEntry.isrStackSize;
    isrStackFree = 0;
    for (idx = debugEntry.pIsrStack; idx < debugEntry.pIsrStack + isrStackSize; idx += 1)
    {
        if (pData[idx] == stackFilled)
        {
            isrStackFree += 1;
        }
        else
        {
            break;
        }
    }

    isrStackUsedPct = (LwF32)(isrStackSize - isrStackFree) / (LwF32)isrStackSize;

    pList = pTaskList;
    while (pList != NULL)
    {
        int size;

        pAddrInfo = &pList->addrInfo;
        size = pAddrInfo->endAddr - pAddrInfo->startAddr + 1;

        switch (pAddrInfo->type)
        {
        case ADDRTYPE_TCB:
            tcbSize += size;
            break;
        case ADDRTYPE_PVTTCB:
            pvtTcbSize += size;
            break;
        case ADDRTYPE_QUEUE:
            queueSize += size;
            break;
        case ADDRTYPE_QUEUESTORAGE:
            queueStorageSize += size;
            break;
        case ADDRTYPE_TASKSTACK:
            taskStackSize += size;
            break;
        case ADDRTYPE_RESIDENT:
            rsdarea = pAddrInfo->endAddr;
            parsedRsdArea = pAddrInfo->startAddr;
            break;
        default:
            dprintf("Unknow addr type!\n");
            break;
        }

        pList = pList->pNext;
    }


    PMU_PRINT_SEPARATOR();
    dprintf("lw:     Data Memory Usage Information\n");
    PMU_PRINT_SEPARATOR();
    dprintf("lw: Dmem Size               = %d bytes\n", dmemSize);
    dprintf("lw: ISR Stack size          = %d bytes\n",isrStackSize);
    dprintf("lw: ISR Stack free          = %d bytes\n",isrStackFree);
    dprintf("lw: ISR Stack usage         = %f %s\n", isrStackUsedPct * 100, "%");
    dprintf("lw: Total tcb size          = %d bytes\n", tcbSize);
    dprintf("lw: Total pvt tcb size      = %d bytes\n", pvtTcbSize);
    dprintf("lw: Total queue size        = %d bytes\n", queueSize);
    dprintf("lw: Total queue store size  = %d bytes\n", queueStorageSize);
    dprintf("lw: Total task stack        = %d bytes\n", taskStackSize);

    if (rsdarea != 0)
    {
        dprintf("lw: Resident area           = %d bytes\n", rsdarea);
        dprintf("lw: Parsed Resident area    = %d bytes\n", parsedRsdArea);
    }
}

/*!
 * Load imem blocks, which will be furhter used to get overlay information
 * Dynamically allocating space to store all the TagBlock structure
 *
 * @param[out]  ppBlocks      Pointer to dynamically allocated TagBlock array
 * @param[out]  pNumBlocks    Number of TagBlocks
 *
 */
static LwBool
loadImemTagBlock
(
    PmuTagBlock **ppBlocks,
    LwU32        *pNumBlocks
)
{
    LwU32        numBlocks    = pPmu[indexGpu].pmuImemGetNumBlocks();
    LwU32        numTagBlocks = 0;
    LwU32        idx;
    PmuTagBlock *pTagBlocks;
    PmuBlock     block;

    for (idx = 0; idx < numBlocks; ++idx)
    {
        if (!(pPmu[indexGpu].pmuImblk(idx, &block)))
        {
            dprintf("ERROR: Failed to get block info %d\n", idx);
            return LW_FALSE;
        }
        if (block.tag > numTagBlocks)
            numTagBlocks = block.tag;
    }

    pTagBlocks = malloc(sizeof(PmuTagBlock) * numTagBlocks);

    if (!pTagBlocks)
    {
        dprintf("ERROR: Unable to allocate memory!\n");
        return LW_FALSE;
    }


    *pNumBlocks = numTagBlocks;
    for (idx = 0; idx < numTagBlocks; ++idx)
    {
        if (!(pPmu[indexGpu].pmuImtag(idx << 8, &pTagBlocks[idx])))
        {
            dprintf("ERROR: Unable to fetch imem block info %d\n", idx);
        }
    }
    *ppBlocks = pTagBlocks;
    return LW_TRUE;
}

/*!
 * Dump IMEM information, overlay loaded information
 *
 * @return   LW_TRUE
 *       Succeeded to load and parse IMEM data
 * @return   LW_FALSE
 *       Failed to retrieve or parse IMEM data
 *
 */
static LwBool
dumpImemInfo (void)
{
    PMU_SYM     *pMatches;
    LwU32        idx;
    LwU32        jdx;
    LwU32        count;
    BOOL         bExactFound;
    const char  *pLoadStartPtn = "_load_start_";
    const char  *pLoadEndPtn = "_load_stop_";
    const char  *pOverlay = "_overlay_id_";
    char         buf[64 + PMU_SYM_NAME_SIZE];
    OverlayInfo *info;
    LwU32        nOverlay;
    PmuTagBlock *pBlocks;
    LwU32        numBlocks;

    if (!loadImemTagBlock(&pBlocks, &numBlocks))
        return LW_FALSE;

    pMatches = pmuSymFind(pLoadStartPtn, FALSE, &bExactFound, &nOverlay);
    if (!pMatches)
    {
        dprintf("ERROR: Unable to find any symbol starts with %s, please update LwWatch!", pLoadStartPtn);
        return LW_FALSE;
    }

    PMU_PRINT_SEPARATOR();
    dprintf("lw:          IMEM Information\n");
    PMU_PRINT_SEPARATOR();

    info = (OverlayInfo *)malloc(sizeof(OverlayInfo) * nOverlay);
    idx = 0;
    while (pMatches != NULL)
    {
        char *subStr;
        info[idx].addrStart = pMatches->addr;
        subStr = strstr(pMatches->name, pLoadStartPtn);
        strncpy(info[idx].overlayName, &subStr[(LwU32)strlen(pLoadStartPtn)],
                PMU_SYM_NAME_SIZE);
        info[idx].overlayName[(LwU32)sizeof(info[idx].overlayName) - 1] = '\0';
        pMatches = pMatches->pTemp;

        idx += 1;
    }

    for (idx = 0; idx < nOverlay; ++idx)
    {
        char targetName[64 + PMU_SYM_NAME_SIZE];

        sprintf(targetName, "%s%s", pLoadEndPtn, info[idx].overlayName);
        pMatches = pmuSymFind(targetName, FALSE, &bExactFound, &count);
        info[idx].addrEnd = 0;

        if (!bExactFound)
        {
            dprintf("ERROR: Unable to find symbol %s, please update LwWatch dumpImemInfo\n", targetName);
            continue;
        }
        info[idx].addrEnd = pMatches->addr;
    }

    dprintf("        Overlay name          Status     Tag Ranges    Total-Blocks    Actual-Loaded\n");
    for (idx = 0; idx < nOverlay; ++idx)
    {
        LwBool bFullLoaded = LW_TRUE;
        LwU32  nLoaded = 0;
        LwU32  nTotal;
        LwU32  nBlkStart = (info[idx].addrStart & ~0xff) >> 8;
        LwU32  nBlkEnd   = (info[idx].addrEnd & ~0xff) >> 8;

        for (jdx = nBlkStart; jdx <= nBlkEnd; ++jdx)
        {
            if (pBlocks[jdx].mapType == PMU_TAG_UNMAPPED)
            {
                bFullLoaded = LW_FALSE;
            }

            if (pBlocks[jdx].mapType != PMU_TAG_UNMAPPED)
            {
                nLoaded += 1;
            }
        }

        sprintf(buf, "%s%s", pOverlay, info[idx].overlayName);
        pMatches = pmuSymFind(buf, FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            if (pMatches->addr == 0)
                continue;
        }

        nTotal = nBlkEnd - nBlkStart + 1;
        if (bFullLoaded)
        {
            dprintf("%-28s   fully     %3d ~ %3d        %2d\n", info[idx].overlayName, nBlkStart, nBlkEnd, nTotal);
        }
        else if (nLoaded > 0)
        {
            dprintf("%-28s   partly    %3d ~ %3d        %2d               %2d\n", info[idx].overlayName, nBlkStart, nBlkEnd, nTotal, nLoaded);
        }
    }

    free(pBlocks);
    free(info);

    return LW_TRUE;
}

static void
dumpQueueInfo
(
    PLinkedList *ppHead
)
{
    PMU_XQUEUE *pQueue;
    LwU32      *pAddrs;
    char       *pQNames;
    LwU32       nQueue;
    LwU32       strSize;
    LwU32       i;

    // Dump queues in summary view
    pmuEventQueueDumpAll(LW_TRUE);

    // Dump queues in details view
    pmuEventQueueDumpAll(LW_FALSE);

    if (!pmuQueueFetchAll(&pQueue, &pAddrs, &pQNames, &strSize, &nQueue))
    {
        dprintf("ERROR: Failed to fetch queue informations, please check LwWatch!\n");
        return;
    }

    for (i = 0; i < nQueue; ++i)
    {
        // Record statistics for queue structure
        PLinkedList pList = (PLinkedList)malloc(sizeof(LinkedList));
        PAddrInfo pAddrInfo = &pList->addrInfo;

        pAddrInfo->startAddr = pAddrs[i];
        pAddrInfo->endAddr = pAddrInfo->startAddr + sizeof(PMU_XQUEUE) - 1;
        pAddrInfo->type = ADDRTYPE_QUEUE;
        sprintf(pAddrInfo->msg, "%s", &pQNames[i * strSize]);
        AddInfoToList(ppHead, pList);

        // Record statistics for queue storage
        pList = (PLinkedList)malloc(sizeof(LinkedList));
        pAddrInfo = &pList->addrInfo;

        pAddrInfo->startAddr = pQueue[i].head;
        pAddrInfo->endAddr = pQueue[i].tail;
        pAddrInfo->type = ADDRTYPE_QUEUESTORAGE;
        sprintf(pAddrInfo->msg, "%s", &pQNames[i * strSize]);
        AddInfoToList(ppHead, pList);
    }

    if (nQueue > 0)
    {
        free(pQueue);
        free(pAddrs);
        free(pQNames);
    }
}

/*!
 * Parse the EXCI Register and dump out the result
 *
 * @param[in]    val32        Value of the EXCI register
 *
 */
static void
parseExciReg(void)
{
    // Parse the value of reg exci, print out all the information we know
    LwU32    excauseMask = 0xf << 20;
    LwU32    expcMask = 0xfffff;
    LwU32    expc;
    PMU_SYM *pMatches;
    PMU_SYM *pSym = NULL;
    LwU32    val32;
    LwU32    excause;

    val32 = pPmu[indexGpu].pmuFalconGetRegister(LW_FALCON_REG_EXCI);
    expc = expcMask & val32;
    excause = (excauseMask & val32) >> 20;

    PMU_PRINT_SEPARATOR();
    dprintf("lw:         EXCI Register Information\n");
    PMU_PRINT_SEPARATOR();
    dprintf("lw:   exci: 0x%08x\n", val32);

    dprintf("lw:   Exception:  ");
    switch (excause)
    {
    case 0:
        dprintf("trap0\n");
        break;
    case 1:
        dprintf("trap1\n");
        break;
    case 2:
        dprintf("trap2\n");
        break;
    case 3:
        dprintf("trap3\n");
        break;
    case 8:
        dprintf("illegal instruction template\n");
        break;
    case 9:
        dprintf("illegal code exelwtion\n");
        break;
    case 10:
        dprintf("imem miss\n");
        break;
    case 11:
        dprintf("imem double hit\n");
        break;
    case 15:
        dprintf("instruction breakpoint\n");
        break;
    default:
        dprintf("Reserved value\n");
        break;
    }

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

      dprintf("lw:   Exception Address:  0x%05x  ", expc);
      if (pSym != NULL)
          dprintf("<%s+0x%x>\n", pSym->name, expc - pSym->addr);
      else
          dprintf("\n");
}

/*!
 * Dump out the full DMEM from 0 to dmemSize
 *
 * @param[in]  pDmem       The buffer storing all DMEM
 * @param[in]  dmemSize    Size of DMEM
 *
 */
static void
dumpDmemContent
(
    LwU8   *pDmem,
    LwU32   dmemSize
)
{
    PMU_PRINT_SEPARATOR();
    dprintf("lw:      Dump out full DMEM\n");
    PMU_PRINT_SEPARATOR();
    printBuffer((char*)pDmem, dmemSize, 0, 0x4);
}

/*!
 * Dump out information about PMU is running or halted
 */
static void
dumpPMURunStatus(void)
{
    PMU_PRINT_SEPARATOR();
    if (pmuHalted())
        dprintf("           PMU is HALTED\n");
    else
    {
        dprintf("           PMU is RUNNING\n");
        dprintf("WARNING: All information below may not be the latest value\n");
        dprintf("         or may be inaclwrate becase PMU is still running!!\n");
    }
    PMU_PRINT_SEPARATOR();
}

/*!
 * Main pmuswak function, print out everything about PMU
 *
 * @param[in]  fileName    log file name, NULL if user don't want to
 *                         write the output to a file.
 * @param[in]  bDetail     Detail view, print out everything including
 *                         the full symbol values, .nm, .objdump and DMEM content.
 *                         By default, those will not be printed out on the screen.
 */
void
pmuswakExec
(
    const char * pFileName,
    LwBool       bDetail
)
{
    LwU32        port         = 1;
    PLinkedList  pHead        = NULL;
    PLinkedList  pTemp;
    LwU8        *pDmem        = NULL;
    LwU32        dmemSize;
    LwU32        uCodeVersion = pPmu[indexGpu].pmuUcodeGetVersion();

    if (!pmuSymCheckAutoLoad())
    {
        return;
    }

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
    if (pFileName != NULL)
    {
        openLogFile(pFileName);
    }
#else
    dprintf("NOTE: Lwrrently only windows supports dump pmuswak log to file.\n");
    dprintf("      For gdb, you can use 'set logging file [filename]' to save\n");
    dprintf("      the output to file.\n");
#endif

    dumpPMURunStatus();
    dumpGPUInfo();
    PMU_PRINT_SEPARATOR();
    dprintf("        AppVersion:  %d\n", uCodeVersion);
    PMU_PRINT_SEPARATOR();

    // Print out all the registers
    pmuExec("r");

    // Parse exci registers
    parseExciReg();

    // Tcb and stacktrace
    dumpTcbAndStacktrace(&pHead);

    // Queue information in brief view and details view
    dumpQueueInfo(&pHead);

    // Memory structure
    dumpMemStructure(pHead);

    // Instruction memory information
    dumpImemInfo();

    // Dump data of all symbols we know about
    loadDmem(&pDmem, &dmemSize, port);
    dumpDmemInfo(pDmem, dmemSize, pHead);

    if (bDetail)
    {
        dumpSymbolContents(pDmem, dmemSize, &pHead);

        // Dump the entire .objdump and .nm file
        dumpSymbolFiles(".objdump");
        dumpSymbolFiles(".nm");
        dumpDmemContent(pDmem, dmemSize);
    }
    free(pDmem);

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
    if (pFileName != NULL)
    {
        closeLogFile();
    }
#else
    dprintf("NOTE: Lwrrently only windows supports dump pmuswak log to file.\n");
    dprintf("      For gdb, you can use 'set logging file [filename]' to save\n");
    dprintf("      the output to file.\n");
#endif

    while (pHead != NULL)
    {
        pTemp = pHead->pNext;
        free(pHead);
        pHead = pTemp;
    }
}
