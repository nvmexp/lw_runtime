/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmuos.c
 * @brief WinDbg Extension for PMU.
 */

/* ------------------------ Includes --------------------------------------- */
#include "hal.h"
#include "pmu.h"

// RM header for command/message interfaces to communicate with PMU
#include "rmpmucmdif.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */

/** @struct  Stores information which will be used to parse event queues
 */
static struct {
    // All the symbols belong to section B
    PMU_SYM **ppEvtqSyms;

    // Number of symbols belong to section B
    LwU32     nEvtqSyms;

    // Address of headPointer of the queue -- "pxQueueListHead"
    // qHead will be set to 0 if symbol pxQueueListHead cannot be found
    LwU32     qHead;

    // Smallest value of field addr for symbols in ppEvtqSyms
    LwU32     addrStart;

    // Largest value of field addr for symbols in ppEvtqSyms
    LwU32     addrEnd;

    // addrEnd - addrStart
    LwU32     memSize;

    // Buffer storing dmem of pmu from addrStart to addrEnd
    LwU8     *pMem;

    // value of symbol _heap
    LwU32     heap;

    // value of symbol _heap_end
    LwU32     heapEnd;
} pmuEvtqInfo;

static LwBool bEvtqSymbolLoaded = LW_FALSE;

/* ------------------------ Function Prototypes ---------------------------- */
static void _pmuXListPrint(LwU32, char *, BOOL);
static void _pmuXQueuePrint(PMU_XQUEUE*, const char *, LwU32, LwU32);
static void _pmuXQueueFullPrint(PMU_XQUEUE*, const char *, LwU32, LwU32);
static void _pmuGetQueueName(LwU32, char *, LwU32);
static BOOL _pmuSymXQueueGetByAddress(LwU32, PMU_XQUEUE *);
static BOOL _pmuEvtqLoadSymbol(void);
static void _pmuEvtqUnloadSymbol(void);

/* ------------------------ Defines ---------------------------------------- */

#define PMU_PRINT_SEPARATOR() \
    dprintf("lw: _________________________________________________________________________________________________\n");

#define TCB_ILWALID_STACKSIZE 0x1
#define TCB_ILWALID_TASKNAME  0x2
#define TCB_ILWALID_TASKID    0x4
#define TCB_ILWALID_STACKTOP  0x8
#define TCB_ILWALID_PRIVPTR   0x10
#define TCB_ILWALID_OVLCNT    0x20

/*!
 * @brief Attempt to get the private TCB from the address provided.
 *
 * It's the caller's responsiblity to free the private tcb being loaded.
 *
 * @param[out]  ppTcbPriv   The function will dynamically allocate a space to store
 *                          the content of private tcb.
 * @param[in]   addr        Address of the private tcb.
 * @param[in]   port        Port to use when reading private TCB address from DMEM.
 *
 * @return 'TRUE'
 *      Private TCB was read successfully.
 *
 * @return 'FALSE'
 *      Failed to get the Private TCB for some reason.
 */
BOOL
pmuTcbGetPriv
(
    PMU_TCB_PVT      **ppTcbPriv,
    LwU32              addr,
    LwU32              port
)
{
    PMU_TCB_PVT *pTmpPriv = NULL;
    PMU_TCB_PVT *pTcbPriv = NULL;
    PMU_SYM     *pMatches = NULL;
    LwU32        actualSize;
    BOOL         bExactFound;
    LwU32        wordsRead;
    LwU32        count;
    LwU32        verNum;
    LwU32        ovlCntImem    = 0;
    LwU32        ovlCntDmem    = 0;
    LwU32        ovlCntImemMax = 0;
    LwU32        ovlCntDmemMax = 0;
    const char  *lwrtosVer = "LwosUcodeVersion";
    const char  *tcbPvtVer= "_PrivTcbVersion";
    BOOL         bResult = FALSE;
    LwU32        tcbPrivSize;

    pMatches = pmuSymFind(lwrtosVer, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        if (!PMU_DEREF_DMEM_PTR_64(pMatches->addr, port, &verNum))
        {
            goto pmuTcbGetPriv_exit;
        }
        verNum = DRF_VAL64(_RM, _PVT_TCB, _VERSION, verNum);
        // When lwrtosVersion was introduced, the PVT_TCB version number was reset.
        // Add 5 to adjust the version number to account for the reset.
        verNum += 0x5;
    }
    else
    {
        // lookup the symbol for the tcb pvt version
        pMatches = pmuSymFind(tcbPvtVer, FALSE, &bExactFound, &count);

        // determine version number based on symbol
        // if the symbol isn't found, revert to version 0
        // else, find out the exact version number and use that
        if (!bExactFound)
        {
            verNum = 0;
        }
        else
        {
            if (!PMU_DEREF_DMEM_PTR(pMatches->addr, port, &verNum))
            {
                goto pmuTcbGetPriv_exit;
            }
        }
    }

    // get the max size for the temporary buffer
    actualSize = sizeof(PMU_TCB_PVT) +
                  (sizeof(LwU8) *
                  (RM_FALC_MAX_ATTACHED_OVERLAYS_IMEM +
                  RM_FALC_MAX_ATTACHED_OVERLAYS_DMEM));

    // malloc for the temporary buffer
    pTmpPriv = (PMU_TCB_PVT *)malloc(actualSize);
    if (pTmpPriv == NULL)
    {
        dprintf("ERROR!!! malloc failed for temporary pvt tcb buffer\n");
        *ppTcbPriv = NULL;
        goto pmuTcbGetPriv_exit;
    }

    tcbPrivSize = sizeof(PMU_TCB_PVT_INT) +
                  (sizeof(LwU8) *
                  (RM_FALC_MAX_ATTACHED_OVERLAYS_IMEM +
                   RM_FALC_MAX_ATTACHED_OVERLAYS_DMEM));
    // dma into temp buffer to get overlay counts
    wordsRead = pPmu[indexGpu].pmuDmemRead(addr,
                                           LW_TRUE,
                                           tcbPrivSize >> 2,
                                           port,
                                           (LwU32*) &(pTmpPriv->pmuTcbPvt));

    // early exit if dma failed
    if (wordsRead != (tcbPrivSize >> 2))
    {
        dprintf("ERROR!!! DMA failed\n");
        *ppTcbPriv = NULL;
        goto pmuTcbGetPriv_exit;
    }

    // use the overlay counts to update the size
    switch (verNum)
    {
        case PMU_TCB_PVT_VER_0:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt;
            ovlCntDmem    = 0;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_0;
            ovlCntDmemMax = 0;
            break;

        case PMU_TCB_PVT_VER_1:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_1;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_1;
            break;

        case PMU_TCB_PVT_VER_2:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_2;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_2;
            break;

        case PMU_TCB_PVT_VER_3:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                            pTmpPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_3;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_3;
            break;

        case PMU_TCB_PVT_VER_4:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                            pTmpPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_4;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_4;
            break;

        case PMU_TCB_PVT_VER_5:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                            pTmpPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_5;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_5;
            break;

        case PMU_TCB_PVT_VER_6:
            ovlCntImem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                            pTmpPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem;
            ovlCntImemMax = PMU_MAX_ATTACHED_OVLS_IMEM_VER_6;
            ovlCntDmemMax = PMU_MAX_ATTACHED_OVLS_DMEM_VER_6;
            break;

        default:
            dprintf("ERROR: %s: Unsupported TCB PVT version number %d\n",
                    __FUNCTION__, verNum);
            goto pmuTcbGetPriv_exit;
    }

    // Early exit if IMEM overlay count is invalid.
    if (ovlCntImem > ovlCntImemMax)
    {
        dprintf("ERROR!!! The number of IMEM overlays in private TCB is larger than %d\n",
                ovlCntImemMax);
        goto pmuTcbGetPriv_exit;
    }

    // Early exit if DMEM overlay count is invalid.
    if (ovlCntDmem > ovlCntDmemMax)
    {
        dprintf("ERROR!!! The number of DMEM overlays in private TCB is larger than %d\n",
                ovlCntDmemMax);
        goto pmuTcbGetPriv_exit;
    }

    // Get updated size based on overlay counts (aligned to 4 bytes).
    actualSize =
        sizeof(PMU_TCB_PVT) + (sizeof(LwU8) * (ovlCntImem + ovlCntDmem));

    // dma again for actual size
    pTcbPriv = (PMU_TCB_PVT *)malloc(actualSize);
    if (pTcbPriv == NULL)
    {
        dprintf("ERROR!!! malloc failed for final pvt TCB buffer\n");
        *ppTcbPriv = NULL;
        goto pmuTcbGetPriv_exit;
    }
    memcpy(pTcbPriv, pTmpPriv, actualSize);

    // update pointer to point to buffer
    *ppTcbPriv = pTcbPriv;

    // update fields outside of the per version private TCB
    (*ppTcbPriv)->tcbPvtVer  = verNum;
    (*ppTcbPriv)->tcbPvtAddr = addr;

    // If we've reached here everything is a OK.
    bResult = TRUE;

    // All other path jumping to this label will return FALSE (failure).
pmuTcbGetPriv_exit:

    // Free temp buffer.
    if (pTmpPriv != NULL)
    {
        free(pTmpPriv);
    }

    return bResult;
}

/*!
 * @brief Attempt to get the address of the current TCB.
 *
 * This is done by looking for the 'pxLwrrentTCB' symbol from the RTOS.
 *
 * @param[out]  pTcb    TCB structure to fill out.
 * @param[in]   port    Port to use when reading current TCB address from DMEM.
 *
 * @return 'TRUE'
 *      Current TCB was read successfully.
 *
 * @return 'FALSE'
 *      Failed to get the current TCB for some reason.
 */
BOOL
pmuTcbGetLwrrent
(
    PMU_TCB *pTcb,
    LwU32    port
)
{
    PMU_SYM *pMatches;
    BOOL     bExactFound;
    LwU32    count;
    LwU32    symValue;

    pMatches = pmuSymFind("pxLwrrentTCB", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        if (PMU_DEREF_DMEM_PTR(pMatches->addr, port, &symValue))
        {
            return pPmu[indexGpu].pmuTcbGet(symValue, port, pTcb);
        }
    }

    return FALSE;
}

/*!
 * @brief Map taskID to task name
 *
 * @param[in]  taskID   The id of the task we are interested in
 *
 * @return     Task Name
 *      Task ID valid
 * @return     !error
 *      Task ID invalid
 */
const char *
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

/*!
 * @brief Retrieve the task name given tcb, regardless what version of the tcb is
 *
 * @param[in]  pTcb        Pointer to the PMU_TCB structure
 * @param[in]  taskName    char buffer to hold the string name
 * @param[in]  bufferSize  size of the buffer, including the trailing \0
 * @param[in]  port        Port used to read dmem
 */
void
pmuGetTaskNameFromTcb
(
    PMU_TCB *pTcb,
    char    *taskName,
    LwU32    bufferSize,
    LwU32    port
)
{
    if (pTcb->tcbVer == PMU_TCB_VER_0)
    {
        LwU32 realSize = (LwU32)strlen(pTcb->pmuTcb.pmuTcb0.taskName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, pTcb->pmuTcb.pmuTcb0.taskName, realSize);
        taskName[realSize] = '\0';
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_1)
    {
        const char* srcName  = pmuGetTaskName((LwU32)pTcb->pmuTcb.pmuTcb1.ucTaskID);
        LwU32       realSize = (LwU32)strlen(srcName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, srcName, realSize);
        taskName[realSize] = '\0';
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_2)
    {
        PMU_TCB_PVT *pPmuTcbPriv = NULL;
        LwU8         taskId;
        const char*  srcName;
        LwU32        realSize;

        //
        // SafeRTOS does not store the task ID in the xTCB.
        // Instead, the task ID is only kept in the private TCB.
        //
        pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb2.pvTcbPvt, 1);
        taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;

        srcName  = pmuGetTaskName(taskId);
        realSize = (LwU32)strlen(srcName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, srcName, realSize);
        taskName[realSize] = '\0';
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_3)
    {
        PMU_TCB_PVT *pPmuTcbPriv = NULL;
        LwU8         taskId;
        const char*  srcName;
        LwU32        realSize;

        pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb3.pvTcbPvt, 1);
        taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;

        srcName  = pmuGetTaskName(taskId);
        realSize = (LwU32)strlen(srcName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, srcName, realSize);
        taskName[realSize] = '\0';
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_4)
    {
        PMU_TCB_PVT *pPmuTcbPriv = NULL;
        LwU8         taskId;
        const char*  srcName;
        LwU32        realSize;

        pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb4.pvTcbPvt, 1);
        switch (pPmuTcbPriv->tcbPvtVer)
        {
            case FLCN_TCB_PVT_VER_0:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID;
                break;
            case FLCN_TCB_PVT_VER_1:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID;
                break;
            case FLCN_TCB_PVT_VER_2:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID;
                break;
            case FLCN_TCB_PVT_VER_3:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID;
                break;
            case FLCN_TCB_PVT_VER_4:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID;
                break;
            case FLCN_TCB_PVT_VER_5:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;
                break;
            case FLCN_TCB_PVT_VER_6:
                taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID;
                break;
            default:
                taskId = -1;
        }

        srcName  = pmuGetTaskName(taskId);
        realSize = (LwU32)strlen(srcName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, srcName, realSize);
        taskName[realSize] = '\0';
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_5)
    {
        PMU_TCB_PVT *pPmuTcbPriv = NULL;
        LwU8         taskId;
        const char*  srcName;
        LwU32        realSize;

        pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb5.pvTcbPvt, 1);
        taskId = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;

        srcName  = pmuGetTaskName(taskId);
        realSize = (LwU32)strlen(srcName);

        if (realSize > bufferSize - 1)
            realSize = bufferSize - 1;

        memcpy(taskName, srcName, realSize);
        taskName[realSize] = '\0';
    }
}

/*!
 * Get the corresponding name of the overlay given the overlay id.
 *
 * @param[in]  id        The overlay id
 * @param[in]  symbols   The overlay id names we have search thorugh pmusym.
 * @param[in]  count     Total count of overlay names.
 *
 * @return     pointer to the string of overlay name
        Overlay ID is valid
   @return     "!error!"
        Overlay ID could not be found
 */
static char *
searchOverlapNameInPmuSym
(
    LwU8      id,
    PMU_SYM  *pSymbols,
    LwU32     count
)
{
    LwU32 index;

    for (index = 0; (index < count) && (pSymbols != NULL); ++index)
    {
        if (pSymbols->addr == (LwU32)id)
            return pSymbols->name;
        pSymbols = pSymbols->pTemp;
    }

    return "!error!";
}

/*!
 * @brief  Validate if the tcb is with correct information
 *
 * @param[in]  pTcb        The tcb to be validated
 *
 * @return     0
 *      The tcb is valid
 * @return     error bitmap (non-zero value)
 *      The tcb is not valid
 */
LwU32
pmuTcbValidate
(
    PMU_TCB *pTcb
)
{
    LwU32        errorCode = 0;
    LwU32        idx;

    //Validate field by field
    if (pTcb->tcbVer == PMU_TCB_VER_0)
    {
        for (idx = 0; idx < sizeof(pTcb->pmuTcb.pmuTcb0.taskName); ++idx)
            if (pTcb->pmuTcb.pmuTcb0.taskName[idx] == '\0')
                break;
        if (idx >= sizeof(pTcb->pmuTcb.pmuTcb0.taskName))
        {
            errorCode |= TCB_ILWALID_TASKNAME;
        }

        if (pTcb->pmuTcb.pmuTcb0.pTopOfStack <= pTcb->pmuTcb.pmuTcb0.pStack)
        {
            errorCode |= TCB_ILWALID_STACKTOP;
        }

        if (pTcb->pmuTcb.pmuTcb0.tcbNumber >= RM_PMU_TASK_ID__END)
        {
            errorCode |= TCB_ILWALID_TASKID;
        }
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_1)
    {
        PMU_TCB_PVT  *pTcbPriv = NULL;
        BOOL          bPriv = pmuTcbGetPriv(&pTcbPriv, pTcb->pmuTcb.pmuTcb1.pvTcbPvt, 1);

        if (bPriv == FALSE)
        {
            errorCode |= TCB_ILWALID_PRIVPTR;
        }
        else
        {
            switch (pTcbPriv->tcbPvtVer)
            {
                case PMU_TCB_PVT_VER_0:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt > PMU_MAX_ATTACHED_OVLS_IMEM_VER_0)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCnt %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_1:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_2:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_3:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_4:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_5:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_6:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID > RM_PMU_TASK_ID__END) ||
                        (pTcb->pmuTcb.pmuTcb1.ucTaskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb1.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                default:
                    dprintf("The version number of PMU TCB PVT is not valid.\n\
                             It's likely that the LwWatch memory is corrupted.\n");
                break;
            }

            free(pTcbPriv);
        }
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_2)
    {
        PMU_TCB_PVT  *pTcbPriv = NULL;
        BOOL          bPriv = pmuTcbGetPriv(&pTcbPriv, pTcb->pmuTcb.pmuTcb2.pvTcbPvt, 1);

        if (bPriv == FALSE)
        {
            errorCode |= TCB_ILWALID_PRIVPTR;
        }
        else
        {
            switch (pTcbPriv->tcbPvtVer)
            {
                case PMU_TCB_PVT_VER_0:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt > PMU_MAX_ATTACHED_OVLS_IMEM_VER_0)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCnt %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_1:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_2:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_3:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_4:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_5:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_6:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb2.pxTopOfStack <=
                        pTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                default:
                    dprintf("The version number of PMU TCB PVT is not valid.\n\
                             It's likely that the LwWatch memory is corrupted.\n");
                break;
            }

            free(pTcbPriv);
        }
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_3)
    {
        PMU_TCB_PVT  *pTcbPriv = NULL;
        BOOL          bPriv = pmuTcbGetPriv(&pTcbPriv, pTcb->pmuTcb.pmuTcb3.pvTcbPvt, 1);

        if (bPriv == FALSE)
        {
            errorCode |= TCB_ILWALID_PRIVPTR;
        }
        else
        {
            switch (pTcbPriv->tcbPvtVer)
            {
                case PMU_TCB_PVT_VER_COUNT:
                    // Satisfy -Werror=switch error
                    break;
                case PMU_TCB_PVT_VER_0:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt > PMU_MAX_ATTACHED_OVLS_IMEM_VER_0)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCnt %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_1:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_2:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_3:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_4:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_5:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_6:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb3.pxTopOfStack <= pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb3.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;
            }
            free(pTcbPriv);
        }
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_4)
    {
        PMU_TCB_PVT  *pTcbPriv = NULL;
        BOOL          bPriv = pmuTcbGetPriv(&pTcbPriv, pTcb->pmuTcb.pmuTcb4.pvTcbPvt, 1);

        if (bPriv == FALSE)
        {
            errorCode |= TCB_ILWALID_PRIVPTR;
        }
        else
        {
            switch (pTcbPriv->tcbPvtVer)
            {
                case PMU_TCB_PVT_VER_COUNT:
                    // Satisfy compiler error about -Werror=switch
                    break;
                case PMU_TCB_PVT_VER_0:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt > PMU_MAX_ATTACHED_OVLS_IMEM_VER_0)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCnt %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_1:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_2:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_3:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_4:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_5:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_6:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb4.pxTopOfStack <= pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb4.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;
            }
            free(pTcbPriv);
        }
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_5)
    {
        PMU_TCB_PVT  *pTcbPriv = NULL;
        BOOL          bPriv = pmuTcbGetPriv(&pTcbPriv, pTcb->pmuTcb.pmuTcb5.pvTcbPvt, 1);

        if (bPriv == FALSE)
        {
            errorCode |= TCB_ILWALID_PRIVPTR;
        }
        else
        {
            switch (pTcbPriv->tcbPvtVer)
            {
                case PMU_TCB_PVT_VER_COUNT:
                    // Satisfy -Werror=switch error
                    break;
                case PMU_TCB_PVT_VER_0:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt > PMU_MAX_ATTACHED_OVLS_IMEM_VER_0)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCnt %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_1:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_1)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_2:
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem > PMU_MAX_ATTACHED_OVLS_IMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_2)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_3:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_3)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_4:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_4)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_5:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_5)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;

                case PMU_TCB_PVT_VER_6:
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                         pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS) > PMU_MAX_ATTACHED_OVLS_IMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid total ovlCntImem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if (pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem > PMU_MAX_ATTACHED_OVLS_DMEM_VER_6)
                    {
                        dprintf("Error validating PMU TCB! Invalid ovlCntDmem %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem);
                        errorCode |= TCB_ILWALID_OVLCNT;
                    }
                    if ((pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID > RM_PMU_TASK_ID__END))
                    {
                        dprintf("Error validating PMU TCB! Invalid task ID %d\n",
                                pTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID);
                        errorCode |= TCB_ILWALID_TASKID;
                    }
                    if (pTcb->pmuTcb.pmuTcb5.pxTopOfStack <= pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress)
                    {
                        dprintf("Error validating PMU TCB! Invalid stack pointer 0x%x\n",
                                pTcb->pmuTcb.pmuTcb5.pxTopOfStack);
                        errorCode |= TCB_ILWALID_STACKTOP;
                    }
                break;
            }
            free(pTcbPriv);
        }
    }
    else
    {
        dprintf("The version number of PMU TCB is not valid.\n\
                 It's likely that the LwWatch memory is corrupted.\n");
    }

    return errorCode;
}


/*!
 * The the contents of the given TCB, the stack pointed to by the TCB, and
 * attempt to callwlate the maximum stack depth that the task associated
 * with the TCB ever realized.
 *
 * @param pTcb[in] Pointer to the TCB to dump
 * @param port[in] Port to use when reading the stack from DMEM
 * @param size[in] Width of each element in the stack dump
 */
void
pmuTcbDump
(
    PMU_TCB *pTcb,
    BOOL     bBrief,
    LwU32    port,
    LwU8     size
)
{
    LwU32           *pStack = NULL;
    LwU32            stackSize = 0;
    LwU8            *pStack8;
    LwU32            maxDepth = 0;
    LwU32            stackSizeBytes;
    LwU32            i;
    BOOL             bExactFound;
    PMU_SYM         *pMatches;
    LwU32            count;
    char             buffer[24];
    PMU_TCB_PVT     *pPmuTcbPriv = NULL;
    LwU32            tcbPStack = 0;
    LwU32            stackTop = 0;
    LwU8             taskId = 0xFF;
    LwF32            stackUsage;
    LwF32            lwrStackUsage;
    LwU32            stackBtm;
    LwU32            wordsRead;

    // Validate the tcb, print out error msg if it is corrupted
    if (pmuTcbValidate(pTcb) != 0)
    {
        dprintf("The tcb structure is corrupted, possibly due to stack overflow.\n");
        dprintf("Please check if stack size of the task is large enough.\n");
        return;
    }

    if (pTcb->tcbVer == PMU_TCB_VER_0)
    {
        stackSize = pTcb->pmuTcb.pmuTcb0.stackDepth;
        tcbPStack = pTcb->pmuTcb.pmuTcb0.pStack;
        stackTop = pTcb->pmuTcb.pmuTcb0.pTopOfStack;
        taskId = (LwU8)pTcb->pmuTcb.pmuTcb0.tcbNumber;
    }
    else
    {
        if (pTcb->tcbVer == PMU_TCB_VER_1)
        {
            pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb1.pvTcbPvt, port);
            stackTop = pTcb->pmuTcb.pmuTcb1.pxTopOfStack;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_2)
        {
            pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb2.pvTcbPvt, port);
            stackTop = pTcb->pmuTcb.pmuTcb2.pxTopOfStack;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_3)
        {
            pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb3.pvTcbPvt, port);
            stackTop = pTcb->pmuTcb.pmuTcb3.pxTopOfStack;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_4)
        {
            pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb4.pvTcbPvt, port);
            stackTop = pTcb->pmuTcb.pmuTcb4.pxTopOfStack;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_5)
        {
            pmuTcbGetPriv(&pPmuTcbPriv, (LwU32)pTcb->pmuTcb.pmuTcb5.pvTcbPvt, port);
            stackTop = pTcb->pmuTcb.pmuTcb5.pxTopOfStack;
        }

        // grab pmu pvt tcb specific variables
        switch (pPmuTcbPriv->tcbPvtVer)
        {
            case PMU_TCB_PVT_VER_0:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID;
                break;

            case PMU_TCB_PVT_VER_1:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID;
                break;

            case PMU_TCB_PVT_VER_2:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID;
                break;

            case PMU_TCB_PVT_VER_3:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID;
                break;

            case PMU_TCB_PVT_VER_4:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID;
                break;

            case PMU_TCB_PVT_VER_5:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;
                break;

            case PMU_TCB_PVT_VER_6:
                stackSize = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.stackSize;
                tcbPStack = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack;
                taskId    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (pTcb->tcbVer == PMU_TCB_VER_3)
        {
            tcbPStack = pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress;
        }
        else if  (pTcb->tcbVer == PMU_TCB_VER_4)
        {
            tcbPStack = pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress;
        }
        if (pTcb->tcbVer == PMU_TCB_VER_5)
        {
            tcbPStack = pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress;
        }
    }

    // read the context of the stack
    pStack = (LwU32 *)malloc(stackSize * sizeof(LwU32));
    if (pStack == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return;
    }
    wordsRead = pPmu[indexGpu].pmuDmemRead(tcbPStack,
                                           LW_TRUE,
                                           stackSize,
                                           port,
                                           pStack);
    if (wordsRead != stackSize)
    {
        dprintf("lw: %s: Unable to read stack at address 0x%x\n",
                __FUNCTION__, tcbPStack);

        stackUsage = -1.0;
        lwrStackUsage = -1.0;
    }
    else
    {
        pStack8        = (LwU8*)pStack;
        stackSizeBytes = stackSize << 2;
        maxDepth = stackSizeBytes;

        for (i = 0; i < stackSizeBytes; i++)
        {
            if (pStack8[i] != 0xa5)
                break;
            maxDepth--;
        }

        stackUsage = (LwF32)maxDepth / (LwF32)(stackSize << 2);
        lwrStackUsage = (LwF32)(tcbPStack + (stackSize << 2) - stackTop) / (LwF32)(stackSize << 2);
    }
    sprintf(buffer, "task%d_", taskId);
    pMatches = pmuSymFind(buffer, FALSE, &bExactFound, &count);
    stackBtm = stackTop + (stackSize << 2);

    // Print out task info
    if (stackTop < tcbPStack)
    {
        dprintf("ERROR:  Stack Overflowed detected\n");
        dprintf("ERROR:  stackTop 0x%x   <   pStack  0x%x\n", stackTop, tcbPStack);
    }

    dprintf("lw:   TCB Address     = 0x%x\n", pTcb->tcbAddr);
    if (pTcb->tcbVer == PMU_TCB_VER_0)
    {
        dprintf("lw:   Task ID         = 0x%x (%s)\n" , pTcb->pmuTcb.pmuTcb0.tcbNumber, pTcb->pmuTcb.pmuTcb0.taskName);
        dprintf("lw:   Priority        = 0x%x\n" , pTcb->pmuTcb.pmuTcb0.priority);
        dprintf("lw:   pStack          = 0x%x\n" , pTcb->pmuTcb.pmuTcb0.pStack);
        dprintf("lw:   stackTop        = 0x%x\n", pTcb->pmuTcb.pmuTcb0.pTopOfStack);
        dprintf("lw:   Stack Size      = %d bytes\n" , (LwU32)(pTcb->pmuTcb.pmuTcb0.stackDepth * sizeof(LwU32)));
    }
    else
    {
        LwU32 taskID = 0;
        LwU32 privilegeLevel = 0;
        LwU32 pStack = 0;
        LwU32 stackSize = 0;
        LwU32 usedHeap = 0;
        LwU32 priority = 0;
        LwU32 topOfStack = 0;
        LwU32 prvTcbAddr = 0;
        LwU32 stackCanary = 0;

        // grab pmu pvt tcb specific variables
        switch (pPmuTcbPriv->tcbPvtVer)
        {
            case PMU_TCB_PVT_VER_0:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.usedHeap;
                break;

            case PMU_TCB_PVT_VER_1:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.usedHeap;
                break;

            case PMU_TCB_PVT_VER_2:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.usedHeap;
                break;

            case PMU_TCB_PVT_VER_3:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.usedHeap;
                break;

            case PMU_TCB_PVT_VER_4:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.usedHeap;
                break;

            case PMU_TCB_PVT_VER_5:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.usedHeap;
                break;

            case PMU_TCB_PVT_VER_6:
                taskID         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.taskID;
                privilegeLevel = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.privilegeLevel;
                pStack         = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.pStack;
                stackSize      = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.stackSize;
                usedHeap       = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.usedHeap;
                stackCanary    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.stackCanary;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (pTcb->tcbVer == PMU_TCB_VER_1)
        {
            priority = pTcb->pmuTcb.pmuTcb1.uxPriority;
            topOfStack = pTcb->pmuTcb.pmuTcb1.pxTopOfStack;
            prvTcbAddr = pTcb->pmuTcb.pmuTcb1.pvTcbPvt;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_2)
        {
            priority = pTcb->pmuTcb.pmuTcb2.ucPriority;
            topOfStack = pTcb->pmuTcb.pmuTcb2.pxTopOfStack;
            prvTcbAddr = pTcb->pmuTcb.pmuTcb2.pvTcbPvt;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_3)
        {
            priority = pTcb->pmuTcb.pmuTcb3.ucPriority;
            topOfStack = pTcb->pmuTcb.pmuTcb3.pxTopOfStack;
            prvTcbAddr = pTcb->pmuTcb.pmuTcb3.pvTcbPvt;
            // Overwrite invalid pStack in modern TCBs
            pStack = pTcb->pmuTcb.pmuTcb3.pcStackBaseAddress;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_4)
        {
            priority = pTcb->pmuTcb.pmuTcb4.uxPriority;
            topOfStack = pTcb->pmuTcb.pmuTcb4.pxTopOfStack;
            prvTcbAddr = pTcb->pmuTcb.pmuTcb4.pvTcbPvt;
            // Overwrite invalid pStack in modern TCBs
            pStack = pTcb->pmuTcb.pmuTcb4.pcStackBaseAddress;
        }
        else if (pTcb->tcbVer == PMU_TCB_VER_5)
        {
            priority = pTcb->pmuTcb.pmuTcb5.ucPriority;
            topOfStack = pTcb->pmuTcb.pmuTcb5.pxTopOfStack;
            prvTcbAddr = pTcb->pmuTcb.pmuTcb5.pvTcbPvt;
            // Overwrite invalid pStack in modern TCBs
            pStack = pTcb->pmuTcb.pmuTcb5.pcStackBaseAddress;
        }

        dprintf("lw:   Task ID         = 0x%x (%s)\n" , taskID, pmuGetTaskName(taskID));
        dprintf("lw:   Priority        = 0x%x\n" , priority);
        dprintf("lw:   Privilege       = 0x%x\n" , privilegeLevel);
        dprintf("lw:   pStack          = 0x%x\n" , pStack);
        dprintf("lw:   Stack Top       = 0x%x\n" , topOfStack);
        dprintf("lw:   Stack Size      = %d bytes\n", (int)(stackSize * sizeof(LwU32)));
        dprintf("lw:   Used Heap       = %d\n", usedHeap);
        dprintf("lw:   PVT TCB Addr    = 0x%x\n", prvTcbAddr);
        dprintf("lw:   TCB Version     = %d\n", pTcb->tcbVer);
        dprintf("lw:   PVT TCB Version = %d\n", pPmuTcbPriv->tcbPvtVer);
        if (pPmuTcbPriv->tcbPvtVer >= PMU_TCB_PVT_VER_6)
        {
            dprintf("lw:   Stack Canary    = 0x%08x\n", stackCanary);
        }
    }

    if (stackUsage >= 0)
    {
        dprintf("lw:   Stack Depth     = %d bytes\n" , maxDepth);
        dprintf("lw:   Stack Usage     = %.2f%%\n", stackUsage * 100.0);
        dprintf("lw:   Lwr Stack Usage = %.2f%%\n", lwrStackUsage * 100.0);

        if (stackUsage == 1.0)
            dprintf("ERROR:   STACK OVERFLOW DETECTED!!!\n");
        else if (stackUsage >= 0.95)
            dprintf("WARNING: The stack is nearly full, considered expanding the stack size.\n");
    }
    else
    {
        dprintf("lw:   Stack is not paged in. Unable to compute stack usage.\n");
    }

    // Print out overlay information if we are using new tcb
    if (pTcb->tcbVer != PMU_TCB_VER_0)
    {
        int ovlCntImem = 0, ovlCntDmem = 0;
        LwU8 *ovlList = NULL;

        switch (pPmuTcbPriv->tcbPvtVer)
        {
            case PMU_TCB_PVT_VER_0:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlCnt;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt0.ovlList;
                break;

            case PMU_TCB_PVT_VER_1:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntImem;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt1.ovlList;
                break;

            case PMU_TCB_PVT_VER_2:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntImem;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt2.ovlList;
                break;

            case PMU_TCB_PVT_VER_3:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemLS +
                             pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntImemHS;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt3.ovlList;
                break;

            case PMU_TCB_PVT_VER_4:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemLS +
                             pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntImemHS;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt4.ovlList;
                break;

            case PMU_TCB_PVT_VER_5:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemLS +
                             pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntImemHS;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt5.ovlList;
                break;

            case PMU_TCB_PVT_VER_6:
                ovlCntImem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemLS +
                             pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntImemHS;
                ovlCntDmem = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlCntDmem;
                ovlList    = pPmuTcbPriv->pmuTcbPvt.pmuTcbPvt6.ovlList;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (ovlCntImem > 0)
        {
            int index;
            PMU_SYM*    pMatches;
            const char* patterlw0 = "_overlay_id_";
            const char* patterlw1 = "_overlay_id_imem_";
            BOOL        bExactFound;
            LwU32       count;

            // The additional +1 is for the additional _ at the begining
            LwU32       offset = (LwU32)strlen(patterlw1) + 1;

            dprintf("lw:   Attached Imem Overlays (max %d):\n", ovlCntImem);
            pMatches = pmuSymFind(patterlw1, FALSE, &bExactFound, &count);

            // if there are no matched symbols for _overlay_id_imem_
            // then check for legacy _overlay_id_ symbols
            if (pMatches == 0)
            {
                offset = (LwU32)strlen(patterlw0) + 1;
                pMatches = pmuSymFind(patterlw0, FALSE, &bExactFound, &count);
            }

            for (index = 0; index < ovlCntImem; ++index)
            {
                LwU8 overlayId = ovlList[index];
                char* overlayName = searchOverlapNameInPmuSym(overlayId, pMatches, count);

                if (overlayName == NULL)
                    dprintf("Unable to find corresponding overlap name for id %x.\n", overlayId);
                else if (overlayId != 0)
                    dprintf("lw:                 0x%02x (%s)\n", overlayId, &overlayName[offset]);
            }

        }

        if (ovlCntDmem > 0)
        {
            int index;
            PMU_SYM*    pMatches;
            const char* pattern = "_overlay_id_dmem_";
            BOOL        bExactFound;
            LwU32       count;

            // The additional +1 is for the additional _ at the begining
            LwU32       offset = (LwU32)strlen(pattern) + 1;

            dprintf("lw:   Attached Dmem Overlays (max %d):\n", ovlCntDmem);
            pMatches = pmuSymFind(pattern, FALSE, &bExactFound, &count);

            for (index = 0; index < ovlCntDmem; ++index)
            {
                LwU8 overlayId = ovlList[ovlCntImem + index];
                char* overlayName = searchOverlapNameInPmuSym(overlayId, pMatches, count);

                if (overlayName == NULL)
                    dprintf("Unable to find corresponding overlap name for id %x.\n", overlayId);
                else if (overlayId != 0)
                    dprintf("lw:                 0x%02x (%s)\n", overlayId, &overlayName[offset]);
            }

        }
    }

    if (pPmuTcbPriv != NULL)
        free(pPmuTcbPriv);

    // Cleanup
    free(pStack);
}

/*!
 * Dump all RTOS scheduler information (ready-lists, suspended list, etc ...)
 */
void
pmuSchedDump
(
    BOOL   bTable
)
{
    PMU_SYM   *pMatches;
    BOOL       bExactFound;
    const char *pReadyTaskListName;
    LwU32      count;
    LwS32      i;
    LwU32      value;


    char *pSimpleLists[] =
    {
        "xSuspendedTaskList"  ,
        "xDelayedTaskList1"   ,
        "xDelayedTaskList2"   ,
    };

    char *pSchedulerVars[] =
    {
        "xSchedulerRunning"   ,
        "uxSchedulerSuspended",
        "uxDmaSuspend"        ,
    };

    if (!pmuSymCheckAutoLoad())
    {
        return;
    }

    //
    // ready task lists - dump in priority order (high-to-low)
    //
    pMatches = pmuSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        LwU64 lwrtosVerNum;
        LwU16 rtosVer;

        if (!PMU_DEREF_DMEM_PTR_64(pMatches->addr, 1, &lwrtosVerNum))
        {
            dprintf("ERROR: Failed to fetch lwrtosVersion variable. \n");
            return;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

        if(rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
        {
            pReadyTaskListName = "xReadyTasksLists";
        }
        else
        {
            pReadyTaskListName = "pxReadyTasksLists";
        }
    }
    else
    {
        pReadyTaskListName = "pxReadyTasksLists";
    }

    count = pMatches->size / sizeof(PMU_XLIST);
    for (i = count - 1; i >= 0; i--)
    {
        char name[32] = {'\0'};
        sprintf(name, "_%s[%d]", pReadyTaskListName, i);

        PMU_PRINT_SEPARATOR();
        _pmuXListPrint(pMatches->addr + (i*sizeof(PMU_XLIST)), name, bTable);
        dprintf("lw:\n");
    }

    //
    // suspended task list
    //

    for (i = 0; i < (LwS32)(sizeof(pSimpleLists)/sizeof(char*)); i++)
    {
        pMatches = pmuSymFind(pSimpleLists[i], FALSE, &bExactFound, &count);
        if ((!bExactFound) || (count != 1))
        {
            return;
        }
        PMU_PRINT_SEPARATOR();
        _pmuXListPrint(pMatches->addr, pMatches->name, bTable);
        dprintf("lw:\n");
    }

    PMU_PRINT_SEPARATOR();
    dprintf("lw:\n");
    dprintf("lw: General:\nlw:\n");

    for (i = 0; i < (LwS32)(sizeof(pSchedulerVars)/sizeof(char*)); i++)
    {
        pMatches = pmuSymFind(pSchedulerVars[i], FALSE, &bExactFound, &count);
        if (count == 1)
        {
            if (PMU_DEREF_DMEM_PTR(pMatches->addr, 1, &value))
            {
                dprintf("lw: [0x%04x] %24s : 0x%08x\n", pMatches->addr, pMatches->name, value);
            }
        }
    }
    return;
}

/*!
 * Fetch tcb stored in the tasksList pointed by the symbol
 *
 * @param[in]  pSymbol      The name of the symbol points to the task list
 * @param[in]  nItem        pSymbol may point to an array, nItem specifies number
 *                          of items in the array, should be 1 for non-array case
 * @param[in]  port         Port used to read dmem
 * @param[out] ppTcbFetched Dynamic allocated memory to hold all tcbs being fetched,
 *                          return the pointer through this variable
 * @param[out] *pTotal      Number of items in the pointer stored in ppTcbFetched
 *
 * @return      LW_TRUE
 *      Fetch operations succeeded
 *              LW_FALSE
 *      Fetch operations failed or memroy allocation failed
 */
static LwBool
_pmuFetchTcbInList
(
    const char *pSymbol,
    LwU32       nItem,
    LwU32       port,
    PMU_TCB   **ppTcbFetched,
    LwU32      *pTotal
)
{
    PMU_TCB  **ppTcbs;
    PMU_TCB   *pTcbs;
    LwU32     *pNTcb;
    PMU_SYM   *pMatches;
    LwU32      count;
    LwU32      addr;
    BOOL       bExactMatch;
    PMU_XLIST  xList;
    LwU32      idx;
    LwU32      total;
    LwU32      i;
    LwBool     bStatus;

    bStatus = LW_TRUE;
    pMatches = pmuSymFind(pSymbol, FALSE, &bExactMatch, &count);

    if (!bExactMatch)
    {
        dprintf("ERROR: Failed to fetch symbol %s, please verify LwWatch and the symbol being loaded\n", pSymbol);
        return LW_FALSE;
    }

    ppTcbs = (PMU_TCB **)malloc(sizeof(PMU_TCB *) * nItem);
    if (ppTcbs == NULL)
    {
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        return LW_FALSE;
    }
    memset(ppTcbs, 0, sizeof(PMU_TCB *) * nItem);

    pNTcb = (LwU32 *)malloc(sizeof(LwU32) * nItem);
    if (pNTcb == NULL)
    {
        free(ppTcbs);
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        return LW_FALSE;
    }

    total = 0;
    addr = pMatches->addr;
    for (idx = 0; idx < nItem; ++idx)
    {
        LwU32 listAddr;
        LwU32 wordsRead;

        wordsRead = pPmu[indexGpu].pmuDmemRead(addr,
                                               LW_TRUE,
                                               sizeof(PMU_XLIST) >> 2,
                                               port,
                                               (LwU32 *)&xList);
        if (wordsRead != (sizeof(PMU_XLIST) >> 2))
        {
            dprintf("ERROR: unable to read list at address 0x%x in function %s\n",
                    addr, __FUNCTION__);
            bStatus = LW_FALSE;
            break;
        }

        pNTcb[idx] = xList.numItems;
        total += xList.numItems;
        ppTcbs[idx] = (PMU_TCB *)malloc(sizeof(PMU_TCB) * xList.numItems);
        if (ppTcbs[idx] == NULL)
        {
            dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
            bStatus = LW_FALSE;
            break;
        }

        listAddr = xList.listEnd.prev;
        for (i = 0; i < xList.numItems; ++i)
        {
            PMU_XLIST_ITEM  item;
            PMU_TCB        *pTcb = &ppTcbs[idx][i];

            wordsRead = pPmu[indexGpu].pmuDmemRead(listAddr,
                                                   LW_TRUE,
                                                   sizeof(item) >> 2,
                                                   port,
                                                   (LwU32 *)&item);
            if (wordsRead != (sizeof(item) >> 2))
            {
                dprintf("ERROR: unable to read list contents at address 0x%x in function %s\n",
                         addr, __FUNCTION__);
                break;
            }

            pPmu[indexGpu].pmuTcbGet(item.owner, port, pTcb);
            listAddr = item.prev;
        }

        addr += sizeof(PMU_XLIST);
    }

    pTcbs = (PMU_TCB *)malloc(sizeof(PMU_TCB) * total);
    if (pTcbs == NULL)
    {
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        bStatus = LW_FALSE;
    }

    i = 0;
    for (idx = 0; idx < nItem; ++idx)
    {
        LwU32 j;

        for (j = 0; j < pNTcb[idx]; ++j)
        {
            if (bStatus)
                memcpy(&pTcbs[i++], &ppTcbs[idx][j], sizeof(PMU_TCB));
        }

        free(ppTcbs[idx]);
    }
    free(ppTcbs);
    free(pNTcb);

    *ppTcbFetched = pTcbs;
    *pTotal = total;

    return bStatus;
}

/*!
 * Fetch all tcbs
 *
 * Note: This function still returns true even partially failure happens.
 *       But an error message will be printed out in this case.
 *
 * @param[in]  port         Port used to read dmem
 * @param[out] ppTcb        Dynamic allocated memory to hold all tcbs being fetched,
 *                          return the pointer through this variable
 * @param[out] *pLen        Number of items in the pointer stored in ppTcbFetched
 *
 * @return      LW_TRUE
 *      Fetch operations succeeded
 *              LW_FALSE
 *      Fetch operations failed or memroy allocation failed
 */
LwBool
pmuTcbFetchAll
(
    LwU32     port,
    PMU_TCB **ppTcb,
    LwU32    *pLen
)
{
    struct
    {
        char             *pSymbol;
        PMU_TCB          *pTcb;
        LwU32             numTcb;
        LwU32             nItem;
    } tcbLists[] = {
                        {NULL, NULL, 0, configMAX_PRIORITIES},
                        {"xDelayedTaskList1",  NULL, 0, 1},
                        {"xDelayedTaskList2",  NULL, 0, 1},
                        {"xPendingReadyList",  NULL, 0, 1},
                        {"xSuspendedTaskList", NULL, 0, 1}
                    };
    LwU32    idx;
    LwU32    jdx;
    LwU32    nItem = sizeof(tcbLists) / sizeof(tcbLists[1]);
    LwU32    total = 0;
    PMU_TCB *pTcb;
    LwBool   bStatus;
    LwBool   bRes = LW_TRUE;
    PMU_SYM *pMatches;
    BOOL     bExactFound;
    LwU32    count;

    pMatches = pmuSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        LwU64 lwrtosVerNum;
        LwU16 rtosVer;

        if (!PMU_DEREF_DMEM_PTR_64(pMatches->addr, port, &lwrtosVerNum))
        {
            dprintf("ERROR: Failed to fetch lwrtosVersion variable. \n");
            return LW_FALSE;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

        if(rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
        {
            tcbLists[0].pSymbol = "xReadyTasksLists";
        }
        else
        {
            tcbLists[0].pSymbol = "pxReadyTasksLists";
        }
    }
    else
    {
        tcbLists[0].pSymbol = "pxReadyTasksLists";
    }

    for (idx = 0; idx < nItem; ++idx)
    {
        bStatus = _pmuFetchTcbInList(tcbLists[idx].pSymbol,
                                     tcbLists[idx].nItem,
                                     port,
                                     &tcbLists[idx].pTcb,
                                     &tcbLists[idx].numTcb);

        if (bStatus)
        {
            total += tcbLists[idx].numTcb;
        }
        else
        {
            bRes = LW_FALSE;
        }
    }

    if ((total == 0) && (!bRes))
    {
        dprintf("ERROR: Failed to fetch tcbs due to errors\n");
        return bRes;
    }
    else if (!bRes)
    {
        dprintf("WARNING: We only fetched part of the tcbs!!!\n");
    }

    bRes = LW_TRUE;
    pTcb = (PMU_TCB *)malloc(sizeof(PMU_TCB) * total);
    if (pTcb == NULL)
    {
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        bRes = LW_FALSE;
        total = 0;
    }

    jdx = 0;

    if (bRes)
    {
        for (idx = 0; idx < nItem; ++idx)
        {
            LwU32 i;

            for (i = 0; i < tcbLists[idx].numTcb; ++i)
                pTcb[jdx++] = tcbLists[idx].pTcb[i];
        }
    }

    for (idx = 0; idx < nItem; ++idx)
        if (tcbLists[idx].pTcb != NULL)
            free(tcbLists[idx].pTcb);

    *ppTcb = pTcb;
    *pLen = total;

    return bRes;
}


/*!
 * Print/dump out information for all PMU TCB's.
 *
 * @param[in]  bBrief  'TRUE' to print a brief message for each TCB; 'FALSE'
 *                     for detailed information.
 */
void
pmuTcbDumpAll
(
    BOOL bBrief
)
{
    LwU32    i;
    PMU_TCB *pTcb;
    LwU32    nTcb;
    char     buf[256];

    if (!pmuSymCheckAutoLoad())
    {
        return;
    }

    if (!pmuTcbFetchAll(1, &pTcb, &nTcb))
    {
        dprintf("ERROR: Failed to fetch tcbs\n");
        return;
    }

    for (i = 0; i < nTcb; ++i)
    {
        dprintf("lw:========================================\n");
        // Validate the tcb, print out error msg if it is corrupted
        if (pmuTcbValidate(&pTcb[i]) != 0)
        {
            pmuGetTaskNameFromTcb(&pTcb[i], buf, sizeof(buf), 1);
            dprintf("*******************************************************************\n");
            dprintf("lw:   !ERROR!  TASK %s\n", buf);
            dprintf("lw:   The tcb strcture is corrupted, possibly due to stack overflow.\n");
            dprintf("lw:   Please check if stack size of the task is large enough.\n");
            dprintf("*******************************************************************\n");
        }
        else
        {
            pmuTcbDump(&pTcb[i], bBrief, 1, 4);
        }
    }
    if (nTcb > 0)
        dprintf("lw:========================================\n");

    free(pTcb);
    return;
}

#undef  offsetof
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

/*!
 * @brief  Unload the cache used to print queues
 */
static void
_pmuEvtqUnloadSymbol(void)
{
    if (bEvtqSymbolLoaded)
    {
        free(pmuEvtqInfo.ppEvtqSyms);
        free(pmuEvtqInfo.pMem);
        bEvtqSymbolLoaded = LW_FALSE;
    }
}

/*!
 * @brief  Load all symbols in section B into struct pmuEvtqInfo
 *
 *
 * @return TRUE
 *      Symbols loaded correctly
 * @return FALSE
 *      Unable to load symbols
 */
static BOOL
_pmuEvtqLoadSymbol(void)
{
    const char  *pcQueueHeader;
    const char  *pcHeap            = "_heap";
    const char  *pcHeapEnd         = "_heap_end";
    PMU_SYM     *pMatches;
    PMU_SYM     *pSymIter;
    BOOL         bExactFound;
    LwU32        count;
    LwU32        i;
    LwU32        wordsRead;

    if (!bEvtqSymbolLoaded)
    {
        if (!pmuSymCheckAutoLoad())
        {
            dprintf("Unable to load .nm file, without which the pmuevtq cannot work, please load .nm with \"pmusym -l [.nmFile]\" manually\n");
            return FALSE;
        }

        // Check for queue list head symbol
        pMatches = pmuSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            LwU64 lwrtosVerNum;
            LwU16 rtosVer;

            if (!PMU_DEREF_DMEM_PTR_64(pMatches->addr, 1, &lwrtosVerNum))
            {
                dprintf("ERROR: Failed to fetch lwrtosVersion variable. \n");
                return FALSE;
            }
            rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

            if(rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
            {
                pcQueueHeader = "xQueueListHead";
            }
            else
            {
                pcQueueHeader = "pxQueueListHead";
            }
        }
        else
        {
            pcQueueHeader = "pxQueueListHead";
        }

        // Check symbol pxQueueListHead
        pMatches = pmuSymFind(pcQueueHeader, FALSE, &bExactFound, &count);
        if (count == 0)
        {
            pmuEvtqInfo.qHead = 0;
        }
        else
        {
            pmuEvtqInfo.qHead = pMatches->addr;
        }

        // Check symbol heap and _heap_end
        pMatches = pmuSymFind(pcHeap, FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            pmuEvtqInfo.heap = pMatches->addr;
        }
        else
        {
            pmuEvtqInfo.heap = 0;
        }

        pMatches = pmuSymFind(pcHeapEnd, FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            pmuEvtqInfo.heapEnd = pMatches->addr;
        }
        else
        {
            pmuEvtqInfo.heap = pPmu[indexGpu].pmuDmemGetSize();
        }

        pMatches = pmuSymFind("_", FALSE, &bExactFound, &count);
        pSymIter = pMatches;
        pmuEvtqInfo.nEvtqSyms = 0;

        while (pSymIter != NULL)
        {
            if ((pSymIter->section != 'T') &&
                (pSymIter->section != 't') &&
                (!pSymIter->bSizeEstimated) &&
                (strstr(pSymIter->name, pcQueueHeader) == NULL)&&
                (pSymIter->addr < pmuEvtqInfo.heapEnd))
            {
                ++pmuEvtqInfo.nEvtqSyms;
            }
            pSymIter = pSymIter->pTemp;
        }

        pmuEvtqInfo.ppEvtqSyms = (PMU_SYM **)malloc(sizeof(PMU_SYM *) * pmuEvtqInfo.nEvtqSyms);

        i = 0;
        pSymIter = pMatches;
        pmuEvtqInfo.addrStart = 0xFFFFFFFF;
        pmuEvtqInfo.addrEnd = 0;
        while (pSymIter != NULL)
        {
            if ((pSymIter->section != 'T') &&
                (pSymIter->section != 't') &&
                (!pSymIter->bSizeEstimated) &&
                (strstr(pSymIter->name, pcQueueHeader) == NULL) &&
                (pSymIter->addr < pmuEvtqInfo.heapEnd))
            {
                pmuEvtqInfo.ppEvtqSyms[i] = pSymIter;
                ++i;

                if (pSymIter->addr < pmuEvtqInfo.addrStart)
                    pmuEvtqInfo.addrStart = pSymIter->addr;

                pmuEvtqInfo.addrEnd = pSymIter->addr + pSymIter->size;
            }
            pSymIter = pSymIter->pTemp;
        }

        if (pmuEvtqInfo.addrEnd <= pmuEvtqInfo.addrStart)
        {
            dprintf("ERROR: Invalid information got from .nm file, please validate the correctness of .nm file.\n");
            return FALSE;
        }

        pmuEvtqInfo.memSize = (pmuEvtqInfo.addrEnd - pmuEvtqInfo.addrStart + sizeof(LwU32) - 1) >> 2 << 2;

        pmuEvtqInfo.pMem = (LwU8 *)malloc(pmuEvtqInfo.memSize);
        if (pmuEvtqInfo.pMem == NULL)
        {
            dprintf("ERROR: Failed to malloc space to cache information for pmuevtq!\n");
            return FALSE;
        }

        wordsRead = pPmu[indexGpu].pmuDmemRead(pmuEvtqInfo.addrStart,
                                               LW_TRUE,
                                               pmuEvtqInfo.memSize >> 2,
                                               1,
                                               (LwU32 *)pmuEvtqInfo.pMem);
        if (wordsRead != (pmuEvtqInfo.memSize >> 2))
        {
            dprintf("ERROR: Unable to read data at address 0x%x\n", pmuEvtqInfo.addrStart);
            return FALSE;
        }

        bEvtqSymbolLoaded = LW_TRUE;
    }

    return bEvtqSymbolLoaded;
}

/*!
 * @brief  Expand the buffer to store the queues, lwrrently will just double
 *         the buffer size
 *
 * @param[out]  ppQueue     Stored all the queues that have been fetched
 * @param[out]  ppAddrs     Stored all the addresses of the queues
 * @param[out]  ppQNames    Stored all the names of the queue in a char array
 * @param[out]  pArraySize  Size of the current array
 * @param[in]   strSize     Size of each queue name (fixed)
 *
 * @return    LW_TRUE
 *      Successfully expand the buffer
 * @return    LW_FALSE
 *      Failed to expand buffer
 */
LwBool
_pmuQueueExpBuf
(
    PMU_XQUEUE **ppQueueArray,
    LwU32      **ppAddrs,
    char       **ppQNameBuf,
    LwU32       *pArraySize,
    LwU32        strSize
)
{
        LwU32 newSize = *pArraySize * 2;
        PMU_XQUEUE *pNewQueue;
        LwU32      *pNewAddr;
        char       *pQNameBuf;

        pNewQueue = (PMU_XQUEUE *)malloc(sizeof(PMU_XQUEUE) * newSize);
        if (pNewQueue == NULL)
        {
            dprintf("ERROR: Out of memory!\n");
            return LW_FALSE;
        }

        pNewAddr = (LwU32 *)malloc(sizeof(PMU_XQUEUE) * newSize);
        if (pNewAddr == NULL)
        {
            dprintf("ERROR: Out of memory!\n");
            return LW_FALSE;
        }

        pQNameBuf = (char *)malloc(sizeof(char) * newSize * strSize);
        if (pNewAddr == NULL)
        {
            dprintf("ERROR: Out of memory!\n");
            return LW_FALSE;
        }

        memcpy(pNewQueue, *ppQueueArray, sizeof(PMU_XQUEUE) * (*pArraySize));
        memcpy(pNewAddr, *ppAddrs, sizeof(LwU32) * (*pArraySize));
        memcpy(pQNameBuf, *ppQNameBuf, sizeof(char) * (*pArraySize) * strSize);

        free(*ppQueueArray);
        free(*ppAddrs);
        free(*ppQNameBuf);

        *ppQueueArray = pNewQueue;
        *ppAddrs = pNewAddr;
        *pArraySize = newSize;

        return LW_TRUE;
}

/*!
 * @brief  Fetch all the queues and return
 *
 * @param[out]  ppQueue     Stored all the queues that have been fetched
 * @param[out]  pAddrs      Stored all the addresses of the queues
 * @param[out]  ppQNames    Stored all the names of the queue in a char array
 * @param[out]  pStrSize    Stored the length of each queue name, so that they
 *                          can be easily fetched from ppQNames
 * @param[out]  pNum        Amount of the queues has been fetched
 *
 * @return LW_TRUE
 *      Ftech queues successfully
 * @return LW_FALSE
 *      Failed to fetch queues
 */
LwBool
pmuQueueFetchAll
(
    PMU_XQUEUE **ppQueue,
    LwU32      **ppAddrs,
    char       **ppQNames,
    LwU32       *pStrSize,
    LwU32       *pNum
)
{
    LwU32       idx;
    LwU32       qAddr;
    LwU32       arraySize;
    PMU_XQUEUE  xQueue;
    PMU_XQUEUE *pQueueArray;
    LwU32      *pAddrs;
    char       *pQNameBuf;
    LwU32       strBuffer;
    LwU32       strIdx;
    LwBool      bLoadedBefore;
    LwU32       wordsRead;

    //
    // NOTE:
    // These strings are used ONLY as a fallback option for old drivers without
    // next pointer in the queue structure, they are subjected to be moved
    // in around 6 months (April/2015).
    //
    const char *pQueueNames[] = {
        "CmdDispQueue"     ,
        "PerfQueue"        ,
        "PcmQueue"         ,
        "PgQueue"          ,
        "PgSmuQueue"       ,
        "Disp2SeqQuque"    ,
        "Disp2I2cQueue"    ,
        "Disp2QGCX"        ,
        "PmgrQueue"        ,
        "Disp2PerfMonQueue",
        "ThermQueue"       ,
        "ThermI2cQueue"    ,
        "Disp2QHdcp"       ,
        "ArcQueue"         ,
        "SpiQueue"         ,
        "CmdQueueMutex"    ,
        "MsgQueueMutex"    ,
        "Timer0Sem"        ,
        "_seqRunSemaphore"
    };

    //
    // If the symbols are loaded when entering this function, then it would not
    // be unloaded during exit, the function who loads the symbol should also
    // be the one who unloads it. So that we can reduce number of times in reading
    // the whole DMEM, which consuming a lot of time.
    //
    bLoadedBefore = bEvtqSymbolLoaded;
    if (!_pmuEvtqLoadSymbol())
    {
        return LW_FALSE;
    }

    arraySize = 32;
    pQueueArray = (PMU_XQUEUE *) malloc(sizeof(PMU_XQUEUE) * arraySize);
    pAddrs = (LwU32 *) malloc(sizeof(LwU32) * arraySize);

    *pStrSize = 64;
    strBuffer = 32 * (*pStrSize);
    pQNameBuf = (char*)malloc(sizeof(char) * strBuffer);

    idx = 0;
    strIdx = 0;

    if (pmuEvtqInfo.qHead != 0)
    {
        wordsRead = pPmu[indexGpu].pmuDmemRead(pmuEvtqInfo.qHead,
                                               LW_TRUE,
                                               1,
                                               1,
                                               &qAddr);
        if (wordsRead != 1)
        {
            dprintf("ERROR: Unable to read data at address 0x%x\n", pmuEvtqInfo.qHead);
            return FALSE;
        }

        while (qAddr != 0)
        {
             if (!_pmuSymXQueueGetByAddress(qAddr, &xQueue))
             {
                break;
             }
             pAddrs[idx] = qAddr;
             pQueueArray[idx] = xQueue;
             _pmuGetQueueName(qAddr, &pQNameBuf[idx * (*pStrSize)], *pStrSize);

             qAddr = xQueue.next;
             ++idx;

             if (idx == arraySize)
             {
                if (!_pmuQueueExpBuf(&pQueueArray, &pAddrs, &pQNameBuf, &arraySize, *pStrSize))
                    break;
             }
        }
    }
    else
    {
        // Fallback option for queue implementation without next pointer
        PMU_SYM *pMatches;
        LwU32    i;
        BOOL     bExactFound;
        LwU32    count;

        for (i = 0; i < sizeof(pQueueNames)/sizeof(char *); i++)
        {
            LwU32 actSize;

            pMatches = pmuSymFind(pQueueNames[i], FALSE, &bExactFound, &count);
            if ((!bExactFound) || (count != 1) || (!PMU_DEREF_DMEM_PTR(pMatches->addr, 1, &qAddr)))
                continue;

             if (!_pmuSymXQueueGetByAddress(qAddr, &xQueue))
             {
                 dprintf("ERROR: The queue structure of %s is corrupted.\n", pQueueNames[i]);
             }

             pAddrs[idx] = qAddr;
             pQueueArray[idx] = xQueue;
             actSize = (LwU32)strlen(pQueueNames[i]);
             actSize = actSize > *pStrSize ? *pStrSize : actSize;
             memcpy(&pQNameBuf[idx * (*pStrSize)], pQueueNames[i], actSize);
             pQNameBuf[(idx + 1) * (*pStrSize) - 1] = '\0';

             ++idx;
             if (idx == arraySize)
             {
                if (!_pmuQueueExpBuf(&pQueueArray, &pAddrs, &pQNameBuf, &arraySize, *pStrSize))
                    break;
             }
        }
    }

    // If the current function loads the symbol, it would be one that unloads it.
    if (!bLoadedBefore)
    {
        _pmuEvtqUnloadSymbol();
    }
    *pNum = idx;
    *ppQueue = pQueueArray;
    *ppAddrs = pAddrs;
    *ppQNames = pQNameBuf;

    return LW_TRUE;
}

/*!
 * Dump information on all known event queues.
 */
void
pmuEventQueueDumpAll
(
    LwBool bSummary
)
{
    LwU32       idx;
    LwU32      *pAddrs;
    PMU_XQUEUE *pQueue;
    char       *pQNames;
    LwU32       nQueue;
    LwU32       strSize;

    PMU_PRINT_SEPARATOR();
    dprintf("lw: PMU Event Queues:\n");
    if (bSummary)
        dprintf("lw: Idx Type    Size       Queue Name    ItmSize  MsgWait   head      tail   readfrom   writeto\n");

    if (!pmuQueueFetchAll(&pQueue, &pAddrs, &pQNames, &strSize, &nQueue))
    {
        dprintf("ERROR: Failed to fetch queue informations, please check LwWatch!\n");
        _pmuEvtqUnloadSymbol();
        return;
    }

    for (idx = 0; idx < nQueue; ++idx)
    {
        if (bSummary)
        {
            _pmuXQueuePrint(&pQueue[idx], &pQNames[strSize * idx], pAddrs[idx], idx);
        }
        else
        {
            PMU_PRINT_SEPARATOR();
            _pmuXQueueFullPrint(&pQueue[idx], &pQNames[strSize * idx], pAddrs[idx], idx);
        }
    }

    if (nQueue > 0)
    {
        free(pQueue);
        free(pAddrs);
        free(pQNames);
    }
}

/*!
 * Dump information on a specific event queue (identified by address).
 *
 * @param[in]  qAddr     Starting address of the queue
 */
void
pmuEventQueueDumpByAddr
(
    LwU32 qAddr
)
{
    PMU_XQUEUE  queue;

    if (!_pmuEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for pmuevtq, with which the function wouldn't work!\n");
        return;
    }

    if (!_pmuSymXQueueGetByAddress(qAddr, &queue))
    {
        dprintf("ERROR: The address 0x%x is not storing a valid queue structure.\n", qAddr);
    }

    PMU_PRINT_SEPARATOR();
    dprintf("lw: PMU Event Queue:\n");
    _pmuXQueueFullPrint(&queue, NULL, qAddr, 0);
    _pmuEvtqUnloadSymbol();

    return;
}

/*!
 * Dump information on a specific event queue (identified by symbol).
 *
 * @param[in]  pSym      Name of the symbol for the queue
 */
void
pmuEventQueueDumpBySymbol
(
    const char *pSym
)
{
    PMU_SYM    *pMatches;
    BOOL        bExactFound;
    LwU32       qAddr;
    LwU32       count;
    PMU_XQUEUE  queue;

    if (!pmuSymCheckAutoLoad())
    {
        dprintf("ERROR: Failed to .nm files, with which the pmuevtq function wouldn't work!\n");
        return;
    }

    if (!_pmuEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for pmuevtq, with which the function wouldn't work!\n");
        return;
    }

    pMatches = pmuSymFind(pSym, FALSE, &bExactFound, &count);
    if (count == 0)
    {
        return;
    }
    if (count > 1)
    {
        dprintf("Error: Muliple symbols found with name <%s>\n", pSym);
        return;
    }

    if (pMatches->addr == 0 || !PMU_DEREF_DMEM_PTR(pMatches->addr, 1, &qAddr))
    {
        dprintf("Error: queue pointer <0x%04x> is invalid.\n", pMatches->addr);
        return;
    }

    if (qAddr == 0)
    {
        dprintf("Error: queue address <0x%04x> is invalid.\n", qAddr);
        return;
    }

    if (!_pmuSymXQueueGetByAddress(qAddr, &queue))
    {
        dprintf("The address 0x%x is not storing a valid queue structure.\n", qAddr);
        return;
    }

    PMU_PRINT_SEPARATOR();
    dprintf("lw: PMU Event Queues:\n");
    _pmuXQueueFullPrint(&queue, &pMatches->name[1], qAddr, 0);
    _pmuEvtqUnloadSymbol();
}

/*!
 * Print an OpenRTOS XLIST in human-readible form
 *
 * @param[in]  listAddr  DMEM address of the list
 * @param[in]  pName     Name for the list being printed since it not contained
 *                       in the list structure itself.
 * @param[in]  bTable    'TRUE' to print in table-form; 'FALSE' to print in
 *                       list-form.
 */
static void
_pmuXListPrint
(
    LwU32  listAddr,
    char  *pName,
    BOOL   bTable
)
{
    PMU_XLIST       xList;
    PMU_XLIST_ITEM  listItem;
    PMU_TCB         tcb;
    LwU32           i;
    LwU32           addr;
    BOOL            bTcbFound;
    char            taskName[64];
    LwU32           wordsRead;

    wordsRead = pPmu[indexGpu].pmuDmemRead(listAddr,
                                           LW_TRUE,
                                           sizeof(PMU_XLIST) >> 2,
                                           1,
                                           (LwU32*)&xList);
    if (wordsRead != (sizeof(PMU_XLIST) >> 2))
    {
        dprintf("ERROR: Unable to read list at address 0x%x\n", listAddr);
        return;
    }

    dprintf("lw: %s\n", pName);
    dprintf("lw:\nlw: List Tracking:\n");
    dprintf("lw:       [0x%04x]   numItems : %d\n"    , listAddr+0x00, xList.numItems);
    dprintf("lw:       [0x%04x]     pIndex : 0x%04x\n", listAddr+0x04, xList.pIndex);
    dprintf("lw:       [0x%04x]  itemValue : 0x%08x\n", listAddr+0x08, xList.listEnd.itemValue);
    dprintf("lw:       [0x%04x]       next : 0x%04x\n", listAddr+0x0c, xList.listEnd.next);
    dprintf("lw:       [0x%04x]       prev : 0x%04x\n", listAddr+0x10, xList.listEnd.prev);

    if (xList.numItems == 0)
        return;

    dprintf("lw:\nlw: Items:\n");

    if (bTable)
    {
        dprintf("lw:       addr:     itemValue:  next:   prev:   owner:  container:  task:\n");
        dprintf("lw:       --------  ----------  ------  ------  ------  ----------  ---------\n");
    }

    addr = xList.listEnd.next;
    for (i = 0; i < xList.numItems; i++)
    {
        wordsRead = pPmu[indexGpu].pmuDmemRead(addr,
                                               LW_TRUE,
                                               sizeof(PMU_XLIST_ITEM) >> 2,
                                               1,
                                               (LwU32*)&listItem);
        if (wordsRead != (sizeof(PMU_XLIST_ITEM) >> 2))
        {
            dprintf("ERROR: Unable to read list item at address 0x%x\n", addr);
            continue;
        }

        bTcbFound = pPmu[indexGpu].pmuTcbGet(listItem.owner, 1, &tcb);
        if (bTcbFound)
        {
            pmuGetTaskNameFromTcb(&tcb, taskName, sizeof(taskName), 1);
        }

        if (bTable)
        {
            dprintf("lw: [%d] : [0x%04x]  0x%08x  0x%04x  0x%04x  0x%04x  0x%04x      <%s>\n",
                    i,
                    addr,
                    listItem.itemValue,
                    listItem.next,
                    listItem.prev,
                    listItem.owner,
                    listItem.container,
                    bTcbFound ? taskName : "");
        }
        else
        {
            dprintf("lw:\n");
            dprintf("lw: [%d] : [0x%04x]  itemValue : 0x%08x\n", i , addr+0x00, listItem.itemValue);
            dprintf("lw:       [0x%04x]       next : 0x%04x\n"     , addr+0x04, listItem.next);
            dprintf("lw:       [0x%04x]       prev : 0x%04x\n"     , addr+0x08, listItem.prev);
            dprintf("lw:       [0x%04x]      owner : 0x%04x <%s>\n", addr+0x0c, listItem.owner, bTcbFound ? taskName : "");
            dprintf("lw:       [0x%04x]  container : 0x%04x\n"     , addr+0x10, listItem.container);
        }

        addr = listItem.next;
    }
    return;
}

#define XLIST_FMT1        "lw:       [0x%04x]%16s : 0x%08x\n"
#define XLIST_FMT2        "lw:       [0x%04x]%16s : 0x%08x    [0x%04x]%7s : 0x%08x\n"
#define XLIST_ITEM_ADDR(base, item) ((LwU32)((base)+offsetof(PMU_XQUEUE, item)))

/*!
 * @brief  Find the name for the queue
 *
 * @param[in]  qAddr              Address of the queue
 * @param[in]  pcQueueNameBuf     Buffer to hold the queue name
 * @param[in]  bufLen             Size of the buffer
 */
static void
_pmuGetQueueName
(
 LwU32  qAddr,
 char  *pcQueueNameBuf,
 LwU32  bufLen
)
{
    BOOL   bVarUnique = TRUE;
    BOOL   bOffsetUnique = TRUE;
    LwU32  targetIdx = pmuEvtqInfo.nEvtqSyms;
    LwU32  offset = 0;
    char   buf[128];
    char  *pcNameChosen;
    LwU32  i;
    LwU32  j;

    if ((!_pmuEvtqLoadSymbol()) || (qAddr == 0))
    {
        pcNameChosen = "!unknown!";
    }
    else
    {
        for (i = 0; i < pmuEvtqInfo.nEvtqSyms; ++i)
        {
            LwU32 pos = pmuEvtqInfo.ppEvtqSyms[i]->addr - pmuEvtqInfo.addrStart;

            for (j = 0; j < pmuEvtqInfo.ppEvtqSyms[i]->size; j += 4, pos += 4)
            {
                if ((*(LwU32*)&pmuEvtqInfo.pMem[pos]) == qAddr)
                {
                    if ((targetIdx != i) && (targetIdx < pmuEvtqInfo.nEvtqSyms))
                    {
                        bVarUnique = FALSE;
                        break;
                    }
                    else if (targetIdx == i)
                    {
                        bOffsetUnique = FALSE;
                        break;
                    }
                    targetIdx = i;
                    offset = j;
                }

                if (!bVarUnique)
                    break;
            }
        }

        if ((bVarUnique) && (targetIdx < pmuEvtqInfo.nEvtqSyms))
        {
            if (pmuEvtqInfo.ppEvtqSyms[targetIdx]->size == 4)
            {
                pcNameChosen = &pmuEvtqInfo.ppEvtqSyms[targetIdx]->name[1];
            }
            else
            {
                if (bOffsetUnique)
                {
                    sprintf(buf, "%s+0x%02x", &pmuEvtqInfo.ppEvtqSyms[targetIdx]->name[1], offset);
                }
                else
                {
                    sprintf(buf, "%s+???", &pmuEvtqInfo.ppEvtqSyms[targetIdx]->name[1]);
                }
                pcNameChosen = buf;
            }
        }
        else
        {
            sprintf(buf, "0x%05x", qAddr);
            pcNameChosen = buf;
        }
    }

    strncpy(pcQueueNameBuf, pcNameChosen, bufLen - 1);
    pcQueueNameBuf[bufLen - 1] = '\0';
}

static void
_pmuXqueuePrintMsg
(
    PMU_XQUEUE *pxQueue,
    LwU32       numSpaces
)
{
    LwU32 wordsRead;

    // Dump the message that is waiting in the queue
    if (pxQueue->messagesWaiting > 0)
    {
        LwU32  idx;
        LwU32  j;
        LwU32  totalSize = pxQueue->length * pxQueue->itemSize;
        LwU32  alignedSize = (totalSize + 3) >> 2 << 2;
        LwU8  *pData = (LwU8 *)malloc(totalSize + alignedSize);
        LwU32  cursor = pxQueue->readFrom - pxQueue->head;
        LwU32  bytesDump;

        wordsRead = pPmu[indexGpu].pmuDmemRead(pxQueue->head,
                                               LW_TRUE,
                                               alignedSize >> 2,
                                               1,
                                               (LwU32 *)pData);

        if (wordsRead == (alignedSize >> 2))
        {
            dprintf("lw:\n");
            for (idx = 0; idx < pxQueue->messagesWaiting; ++idx)
            {
                dprintf("lw:");
                for (j = 0; j < numSpaces; ++j)
                    dprintf(" ");
                dprintf("Msg%02d  0x%05x:  ", idx, cursor + pxQueue->head);

                for (bytesDump = 0; bytesDump < pxQueue->itemSize; ++bytesDump)
                {
                    if ((bytesDump & 3) == 0)
                        dprintf(" ");
                    dprintf("%02x", pData[cursor++]);
                    if (cursor > totalSize - 1)
                        cursor = 0;

                    if ((bytesDump != 0) && (bytesDump < pxQueue->itemSize - 1)
                        && ((bytesDump & 0xf) == 0xf))
                    {
                        dprintf("\nlw:");
                        for (j = 0; j < numSpaces; ++j)
                            dprintf(" ");
                        dprintf("       0x%05x:  ", cursor + pxQueue->head);
                    }
                }
                dprintf("\n");
            }
            dprintf("lw:\n");
        }
        else
        {
            dprintf("ERROR: Unable to read data at address 0x%x\n", pxQueue->head);
        }
        free(pData);
    }

}

static void
_pmuXQueueFullPrint
(
    PMU_XQUEUE *pxQueue,
    const char *pcQueueName,
    LwU32       qAddr,
    LwU32       queueIdx
)
{
    dprintf("\nlw: Queue %d: %s\n\n", queueIdx, pcQueueName);
    dprintf(XLIST_FMT2, XLIST_ITEM_ADDR(qAddr, messagesWaiting), "messagesWaiting", pxQueue->messagesWaiting,
                        XLIST_ITEM_ADDR(qAddr, head)           , "head"           , pxQueue->head);
    dprintf(XLIST_FMT2, XLIST_ITEM_ADDR(qAddr, length)         , "length"         , pxQueue->length         ,
                        XLIST_ITEM_ADDR(qAddr, tail)           , "tail"           , pxQueue->tail);
    dprintf(XLIST_FMT2, XLIST_ITEM_ADDR(qAddr, itemSize)       , "itemSize"       , pxQueue->itemSize       ,
                        XLIST_ITEM_ADDR(qAddr, rxLock)         , "rxLock"         , pxQueue->rxLock);
    dprintf(XLIST_FMT2, XLIST_ITEM_ADDR(qAddr, writeTo)        , "writeTo"        , pxQueue->writeTo        ,
                        XLIST_ITEM_ADDR(qAddr, txLock)         , "txLock"         , pxQueue->txLock);
    dprintf(XLIST_FMT1, XLIST_ITEM_ADDR(qAddr, readFrom)       , "readFrom"       , pxQueue->readFrom);
    dprintf("lw:\n");

    _pmuXListPrint(qAddr+offsetof(PMU_XQUEUE, xTasksWaitingToSend),
                   "Waiting-To-Send List:",
                    TRUE);
    dprintf("lw:\n");
    _pmuXListPrint(qAddr+offsetof(PMU_XQUEUE, xTasksWaitingToReceive),
                   "Waiting-To-Receive List:",
                    TRUE);
    _pmuXqueuePrintMsg(pxQueue, 7);

}

/*!
 * Print out and XQUEUE structure in human-readible form.
 *
 * @param[in]  pxQueue       The queue structure to be printed out
 * @param[in]  pcQueueName   Name of the queue
 * @param[in]  qAddr         Device address of the queue
 * @param[in]  queueIdx      Index of the queue
 */
static void
_pmuXQueuePrint
(
    PMU_XQUEUE *pxQueue,
    const char *pcQueueName,
    LwU32       qAddr,
    LwU32       queueIdx
)
{
    const char *pcQueueType;
    char        queueNameBuf[128];

    if (pcQueueName == NULL)
    {
        _pmuGetQueueName(qAddr, queueNameBuf, sizeof(queueNameBuf));
        pcQueueName = queueNameBuf;
    }

    //
    // Print out format
    // queueIndex  queueType(Q--Queue/S--Semaphore)  totalSizeOfQueue/TakenOrNotTaken(Semaphore) VariableName
    // head  tail  readfrom writeto  messageWaiting (For semaphore, items of this line will be omitted)
    //
    if (pxQueue->itemSize == 0)
    {
        const char *pcSemphrStatus;

        if (pxQueue->messagesWaiting == 0)
            pcSemphrStatus = "Taken";
        else
            pcSemphrStatus = "Given";

        pcQueueType = "S";
        dprintf("lw: %3d   %s   %s   %17s\n",
                queueIdx,
                pcQueueType,
                pcSemphrStatus,
                pcQueueName
                );

    }
    else
    {
        pcQueueType = "Q";

        dprintf("lw: %3d   %s   %5d   %17s   %6d   %4d   0x%05x   0x%05x   0x%05x   0x%05x\n",
                queueIdx,
                pcQueueType,
                pxQueue->length,
                pcQueueName,
                pxQueue->itemSize,
                pxQueue->messagesWaiting,
                pxQueue->head,
                pxQueue->tail,
                pxQueue->readFrom,
                pxQueue->writeTo
                );

        // Dump the message that is waiting in the queue
        _pmuXqueuePrintMsg(pxQueue, 23);
    }
}

/*!
 * Read in an XQUEUE structure starting at DMEM address 'addr'. Also performs
 * basic structural validation on data for sanity.
 *
 * @param[in]   addr     Starting address of the XQUEUE structure to read
 * @param[out]  pXQueue  Pointer to the XQUEUE structure to populate
 *
 * @return 'TRUE'   if that read into the structure can be trusted
 * @return 'FALSE'  upon error
 */
static BOOL
_pmuSymXQueueGetByAddress
(
    LwU32       addr,
    PMU_XQUEUE *pXQueue
)
{
    LwU32 wordsRead;
    LwU32 length = sizeof(PMU_XQUEUE) >> 2;
    LwU32 heapLowest;
    LwU32 heapHighest;
    LwU32 maxMsg;

    if (!_pmuEvtqLoadSymbol())
    {
        heapLowest = 0;
        heapHighest = pPmu[indexGpu].pmuDmemGetSize();

        //
        // Set the maximum value to 0xFF which seems to be
        // an impoosible value to reach, but REMEMBER to update
        // if we go more than that
        maxMsg = 0xFF;
    }
    else
    {
        heapLowest = pmuEvtqInfo.heap;
        heapHighest = pmuEvtqInfo.heapEnd;
        maxMsg = heapHighest - heapLowest;
    }

    // read-in the data structure
    wordsRead = pPmu[indexGpu].pmuDmemRead(addr,             // addr
                                           LW_TRUE,          // addr is VA
                                           length,           // length
                                           1,                // port
                                           (LwU32*)pXQueue); // pDmem
    if (wordsRead != length)
    {
        return FALSE;
    }

    // do some basic validation
    if ((pXQueue->head            >=  heapHighest)    ||
        (pXQueue->head            <   heapLowest)     ||
        (pXQueue->tail            >=  heapHighest)    ||
        (pXQueue->tail            <   heapLowest)     ||
        (pXQueue->tail            <   pXQueue->head)  ||
        (pXQueue->readFrom        <   pXQueue->head)  ||
        (pXQueue->writeTo         >   pXQueue->tail)  ||
        (pXQueue->messagesWaiting >=  maxMsg)
       )
    {
        return FALSE;
    }
    return TRUE;
}
