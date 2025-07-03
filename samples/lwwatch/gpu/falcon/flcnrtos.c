/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "falcon.h"

extern POBJFLCN thisFlcn;

/* ------------------------ Static variables ------------------------------- */

/** @struct  Stores information which will be used to parse event queues
 */
static struct {
    // All the symbols belong to section B
    FLCN_SYM **ppEvtqSyms;

    // Number of symbols belong to section B
    LwU32      nEvtqSyms;

    // Address of headPointer of the queue -- "pxQueueListHead"
    // qHead will be set to 0 if symbol pxQueueListHead cannot be found
    LwU32      qHead;

    // Smallest value of field addr for symbols in ppEvtqSyms
    LwU32      addrStart;

    // Largest value of field addr for symbols in ppEvtqSyms
    LwU32      addrEnd;

    // addrEnd - addrStart
    LwU32      memSize;

    // Buffer storing dmem of flcn from addrStart to addrEnd
    LwU8      *pMem;

    // value of symbol _heap
    LwU32      heap;

    // value of symbol _heap_end
    LwU32      heapEnd;
} flcnEvtqInfo;

static LwBool bEvtqSymbolLoaded = LW_FALSE;

/* ------------------------ Function Prototypes ---------------------------- */

// Internal helper routines
static void   _XListPrint(LwU32, char *, BOOL);
static void   _flcnGetQueueName(LwU32, char *, LwU32);
static void   _flcnXQueuePrint(FLCN_RTOS_XQUEUE *, const char *, LwU32, LwU32);
static void   _flcnXQueueFullPrint(FLCN_RTOS_XQUEUE *, const char *, LwU32, LwU32);
static BOOL   _flcnSymXQueueGetByAddress(LwU32, FLCN_RTOS_XQUEUE *);
static LwBool _flcnQueueExpBuf(FLCN_RTOS_XQUEUE **, LwU32 **, char **, LwU32 *, LwU32);
static char * _searchOverlapNameInFlcnSym(LwU8, FLCN_SYM *, LwU32);
static LwBool _flcnFetchTcbInList(const char *, LwU32, LwU32, FLCN_RTOS_TCB **, LwU32 *);
static void   _flcnEvtqUnloadSymbol(void);
static BOOL   _flcnEvtqLoadSymbol(void);
static void   _flcnGetQueueName(LwU32, char *, LwU32);
static LwBool _flcnQueueExpBuf(FLCN_RTOS_XQUEUE **, LwU32 **, char **, LwU32 *, LwU32);
static void   _flcnGetQueueName(LwU32, char *, LwU32);
static void   _flcnXqueuePrintMsg(FLCN_RTOS_XQUEUE *, LwU32);
static void   _flcnXQueueFullPrint(FLCN_RTOS_XQUEUE *, const char *, LwU32, LwU32);
static void   _flcnXQueuePrint(FLCN_RTOS_XQUEUE *, const char *, LwU32, LwU32);
static BOOL   _flcnSymXQueueGetByAddress(LwU32, FLCN_RTOS_XQUEUE *);
static LwBool _flcnGetQueueIdToTaskIdMapAddrAndSize(LwU32 *pAddr, LwU32 *pSize);
static LwBool _flcnReadQueueIdToTaskIdMap(LwU8 **ppUcodeQueueIdToTaskIdMap, LwU32 mapAddr, LwU32 mapSizeBytes);
static LwBool _flcnGetQueueIdToTaskIdMap(LwU8 **ppUcodeQueueIdToTaskIdMap, LwU8 *pNumElements);
static LwBool _flcnGetQueueIdFromTaskId(LwU32 qTaskId, LwU8 *pQueueId);
static LwBool _flcnGetQueueAddrFromQueueId(LwU8 queueId, LwU32 *pQueueAddr);

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
flcnTcbGetPriv
(
    FLCN_TCB_PVT  **ppTcbPriv,
    LwU32              addr,
    LwU32              port
)
{
    FLCN_TCB_PVT *pTmpPriv = NULL;
    FLCN_TCB_PVT *pTcbPriv = NULL;
    FLCN_SYM     *pMatches = NULL;
    LwBool        bExactFound;
    LwBool        bResult = LW_FALSE;
    LwU32         actualSize;
    LwU32         count;
    LwU32         verNum;
    LwU64         lwrtosVerNum;
    LwU32         ovlCntImem    = 0;
    LwU32         ovlCntDmem    = 0;
    LwU32         ovlCntImemMax = 0;
    LwU32         ovlCntDmemMax = 0;
    LwU32         engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();
    const char   *lwrtosVer = "LwosUcodeVersion";
    const char   *tcbPvtVer= "_PrivTcbVersion";
    LwU32         tcbPrivSize;
    LwU32         wordsRead;

    pMatches = flcnSymFind(lwrtosVer, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        if (!FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, pMatches->addr, port, &lwrtosVerNum))
        {
            goto flcnTcbGetPriv_exit;
        }
        verNum = DRF_VAL64(_RM, _PVT_TCB, _VERSION, lwrtosVerNum);
        // When lwrtosVersion was introduced, the PVT_TCB version number was reset.
        // Add 5 to adjust the version number to account for the reset.
        verNum += 0x5;
    }
    else
    {
        // lookup the symbol for the tcb pvt version
        pMatches = flcnSymFind(tcbPvtVer, FALSE, &bExactFound, &count);

        // determine version number based on symbol
        // if the symbol isn't found, revert to version 0
        // else, find out the exact version number and use that
        if (!bExactFound)
        {
            verNum = 0;
        }
        else
        {
            if (!FLCN_RTOS_DEREF_DMEM_PTR(engineBase, pMatches->addr, port, &verNum))
            {
                goto flcnTcbGetPriv_exit;
            }
        }
    }

    // dma into temp buffer to get overlay counts
    actualSize = sizeof(FLCN_TCB_PVT) +
                  (sizeof(LwU8) *
                  (RM_FALC_MAX_ATTACHED_OVERLAYS_IMEM +
                  RM_FALC_MAX_ATTACHED_OVERLAYS_DMEM));

    // malloc for the temporary buffer
    pTmpPriv = (FLCN_TCB_PVT *)malloc(actualSize);
    if (pTmpPriv == NULL)
    {
        dprintf("ERROR!!! malloc failed for temporary pvt tcb buffer\n");
        *ppTcbPriv = NULL;
        goto flcnTcbGetPriv_exit;
    }

    tcbPrivSize = sizeof(FLCN_TCB_PVT_INT) +
                  (sizeof(LwU8) *
                  (RM_FALC_MAX_ATTACHED_OVERLAYS_IMEM +
                  RM_FALC_MAX_ATTACHED_OVERLAYS_DMEM));

    // dma into temp buffer to get overlay counts
    wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                             addr,
                                             LW_TRUE,
                                             tcbPrivSize >> 2,
                                             port,
                                             (LwU32*) &(pTmpPriv->flcnTcbPvt));

    // early exit if dma failed
    if (wordsRead != (tcbPrivSize >> 2))
    {
        dprintf("ERROR!!! DMA failed\n");
        *ppTcbPriv = NULL;
        goto flcnTcbGetPriv_exit;
    }

    // use the overlay counts to update the size
    switch (verNum)
    {
        case FLCN_TCB_PVT_VER_0:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt0.ovlCnt;
            ovlCntDmem    = 0;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_0;
            ovlCntDmemMax = 0;
            break;

        case FLCN_TCB_PVT_VER_1:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt1.ovlCntImem;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt1.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_1;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_1;
            break;

        case FLCN_TCB_PVT_VER_2:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt2.ovlCntImem;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt2.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_2;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_2;
            break;

        case FLCN_TCB_PVT_VER_3:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntImemLS +
                            pTmpPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_3;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_3;
            break;

        case FLCN_TCB_PVT_VER_4:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntImemLS +
                            pTmpPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_4;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_4;
            break;

        case FLCN_TCB_PVT_VER_5:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntImemLS +
                            pTmpPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_5;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_5;
            break;

        case FLCN_TCB_PVT_VER_6:
            ovlCntImem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntImemLS +
                            pTmpPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntImemHS;
            ovlCntDmem    = pTmpPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntDmem;
            ovlCntImemMax = FLCN_MAX_ATTACHED_OVLS_IMEM_VER_6;
            ovlCntDmemMax = FLCN_MAX_ATTACHED_OVLS_DMEM_VER_6;
            break;

        default:
            dprintf("ERROR: %s: Unsupported TCB PVT version number %d\n",
                    __FUNCTION__, verNum);
            goto flcnTcbGetPriv_exit;
    }

    // Early exit if IMEM overlay count is invalid.
    if (ovlCntImem > ovlCntImemMax)
    {
        dprintf("ERROR!!! The number of IMEM overlays in private TCB is larger than %d\n",
                ovlCntImemMax);
        goto flcnTcbGetPriv_exit;
    }

    // Early exit if DMEM overlay count is invalid.
    if (ovlCntDmem > ovlCntDmemMax)
    {
        dprintf("ERROR!!! The number of DMEM overlays in private TCB is larger than %d\n",
                ovlCntDmemMax);
        goto flcnTcbGetPriv_exit;
    }

    // Get updated size based on overlay counts (aligned to 4 bytes).
    actualSize =
        sizeof(FLCN_TCB_PVT) + (sizeof(LwU8) * (ovlCntImem + ovlCntDmem));

    // dma again for actual size
    pTcbPriv = (FLCN_TCB_PVT *)malloc(actualSize);
    if (pTcbPriv == NULL)
    {
        dprintf("ERROR!!! malloc failed for final pvt TCB buffer\n");
        *ppTcbPriv = NULL;
        goto flcnTcbGetPriv_exit;
    }
    memcpy(pTcbPriv, pTmpPriv, actualSize);

    // update pointer to point to buffer
    *ppTcbPriv = pTcbPriv;

    // update fields outside of the per version private TCB
    (*ppTcbPriv)->tcbPvtVer  = verNum;
    (*ppTcbPriv)->tcbPvtAddr = addr;

    // If we've reached here everything is a OK.
    bResult = LW_TRUE;

flcnTcbGetPriv_exit:

    // Free temp buffer.
    if (pTmpPriv != NULL)
    {
        free(pTmpPriv);
    }

    return bResult;
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
flcnTcbValidate
(
    FLCN_RTOS_TCB *pTcb
)
{
    LwU32 errorCode = 0;
    return errorCode;
}

/*!
 * Using the provided TCB address, goes and reads DMEM to populate the given
 * TCB structure. thisFlcn object should be properly set by the caller
 * to point to the target falcon with RTOS loaded.
 *
 * @param[in]   tcbAddress   DMEM address of the TCB to retrieve
 * @param[in]   port         Port to use when reading the DMEM
 * @param[out]  pTcb         Pointer to the TCB structure to populate
 *
 * @return TRUE if the TCB was retrieved without error; FALSE otherwise
 */
BOOL
flcnRtosTcbGet
(
    LwU32           tcbAddress,
    LwU32           port,
    FLCN_RTOS_TCB   *pTcb
)
{
    LwU32       size = 0;
    LwU32       engineBase = 0x0;
    LwU32      *pBuffer = NULL;
    FLCN_SYM   *pMatches;
    LwBool      bExactFound;
    LwU32       count;
    LwU64       lwrtosVerNum;
    const char *pSymLwrtosVersion = "LwosUcodeVersion";
    const char *pSymOsDebugEntryPoint = "OsDebugEntryPoint";

    LwU32       wordsRead;
    BOOL        bSuccess = TRUE;

    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    pMatches = flcnSymFind(pSymLwrtosVersion, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        LwU16 rtosVer;
        LwU8  tcbVer;

        if (!FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, pMatches->addr, port, &lwrtosVerNum))
        {
            dprintf("lw: %s: Unable to retrieve lwrtosVersion symbol.\n", __FUNCTION__);
            return FALSE;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS, _VERSION, lwrtosVerNum);
        tcbVer = DRF_VAL64(_RM, _TCB, _VERSION, lwrtosVerNum);

        if (rtosVer == LW_RM_RTOS_VERSION_SAFERTOS_V5160_LW12_FALCON)
        {
            if (0x0 == tcbVer)
            {
                // Original TCB version when defines were set for v5.16.0-lw1.2
                pTcb->tcbVer = FLCN_TCB_VER_2;
                size = (sizeof(pTcb->flcnTcb.flcnTcb2) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else if (0x1 == tcbVer)
            {
                pTcb->tcbVer = FLCN_TCB_VER_3;
                size = (sizeof(pTcb->flcnTcb.flcnTcb3) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else if (0x2 == tcbVer)
            {
                pTcb->tcbVer = FLCN_TCB_VER_5;
                size = (sizeof(pTcb->flcnTcb.flcnTcb5) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else
            {
                dprintf("lw: %s: Invalid LW_RM_TCB_VERSION.\n", __FUNCTION__);
                return FALSE;
            }
        }
        if (rtosVer == LW_RM_RTOS_VERSION_SAFERTOS_V5160_LW13_FALCON)
        {
            if (0x0 == tcbVer)
            {
                pTcb->tcbVer = FLCN_TCB_VER_5;
                size = (sizeof(pTcb->flcnTcb.flcnTcb5) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else
            {
                dprintf("lw: %s: Invalid LW_RM_TCB_VERSION.\n", __FUNCTION__);
                return FALSE;
            }
        }
        else if (rtosVer == LW_RM_RTOS_VERSION_OPENRTOS_V413_LW10_FALCON)
        {
            if (0x0 == tcbVer)
            {
                pTcb->tcbVer = FLCN_TCB_VER_1;
                size = (sizeof(pTcb->flcnTcb.flcnTcb1) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else if (0x1 == tcbVer)
            {
                pTcb->tcbVer = FLCN_TCB_VER_4;
                size = (sizeof(pTcb->flcnTcb.flcnTcb4) + sizeof(LwU32)) / sizeof(LwU32);
            }
        }
        else
        {
            dprintf("lw: Unknown RTOS version %d.\n", rtosVer);
        }
    }
    else
    {
        //
        // We need to know we are doing the old or the new driver
        // The main difference is if we have a private tcb structure to hold the information
        // By checking the symbol OsDebugEntryPoint, we can know if it is new.
        //
        pMatches = flcnSymFind(pSymOsDebugEntryPoint, FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            pTcb->tcbVer = FLCN_TCB_VER_1;
            size = (sizeof(pTcb->flcnTcb.flcnTcb1) + sizeof(LwU32)) / sizeof(LwU32);
        }
        else
        {
            pTcb->tcbVer = FLCN_TCB_VER_0;
            size = (sizeof(pTcb->flcnTcb.flcnTcb0) + sizeof(LwU32)) / sizeof(LwU32);
        }
    }

    // Create a temporary buffer to store data
    pBuffer = (LwU32 *)malloc(size * sizeof(LwU32));
    if (pBuffer == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return FALSE;
    }

    // Read the raw TCB data from the DMEM
    wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              tcbAddress,
                                              LW_TRUE,
                                              size,
                                              port,
                                              pBuffer);

    if (wordsRead == size)
    {
        pTcb->tcbAddr = tcbAddress;
        if (pTcb->tcbVer == FLCN_TCB_VER_0)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb0, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb0));
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_1)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb1, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb1));
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_2)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb2, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb2));
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_3)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb3, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb3));
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_4)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb4, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb4));
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_5)
        {
            memcpy(&pTcb->flcnTcb.flcnTcb5, pBuffer, sizeof(pTcb->flcnTcb.flcnTcb5));
        }
    }
    else
    {
        dprintf("lw: %s: unable to read DMEM at address 0x%x\n",
                __FUNCTION__, tcbAddress);
        bSuccess = FALSE;
    }

    free(pBuffer);
    return bSuccess;
}

/*!
 * @brief  Given the tcb structure, return the corresponding task name
 *
 * @param[in]   pTcb   The tcb we want its name
 *
 * @return      const char * string, NULL if tcb's version is unknown
 */
const char *
flcnRtosGetTasknameFromTcb
(
    FLCN_RTOS_TCB * pTcb
)
{
    if (pTcb->tcbVer == FLCN_TCB_VER_0)
    {
        return pTcb->flcnTcb.flcnTcb0.taskName;
    }
    else if (pTcb->tcbVer == FLCN_TCB_VER_1)
    {
        return thisFlcn->pFCIF->flcnGetTasknameFromId(pTcb->flcnTcb.flcnTcb1.ucTaskID);
    }
    else if (pTcb->tcbVer == FLCN_TCB_VER_2)
    {
        FLCN_TCB_PVT *pFlcnTcbPriv = NULL;
        LwU8 taskId;

        //
        // SafeRTOS does not have the task ID in the xTCB. Instead, use the
        // task ID stored in the private TCB.
        //
        flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb2.pvTcbPvt, 1);
        taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
        return thisFlcn->pFCIF->flcnGetTasknameFromId(taskId);
    }
    else if (pTcb->tcbVer == FLCN_TCB_VER_3)
    {
        FLCN_TCB_PVT *pFlcnTcbPriv = NULL;
        LwU8 taskId;

        //
        // SafeRTOS does not have the task ID in the xTCB. Instead, use the
        // task ID stored in the private TCB.
        //
        flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb3.pvTcbPvt, 1);
        taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
        return thisFlcn->pFCIF->flcnGetTasknameFromId(taskId);
    }
    else if (pTcb->tcbVer == FLCN_TCB_VER_4)
    {
        FLCN_TCB_PVT *pFlcnTcbPriv = NULL;
        LwU8 taskId;

        //
        // SafeRTOS does not have the task ID in the xTCB. Instead, use the
        // task ID stored in the private TCB.
        //
        flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb4.pvTcbPvt, 1);
        switch (pFlcnTcbPriv->tcbPvtVer)
        {
            case FLCN_TCB_PVT_VER_0:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.taskID;
                break;
            case FLCN_TCB_PVT_VER_1:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.taskID;
                break;
            case FLCN_TCB_PVT_VER_2:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.taskID;
                break;
            case FLCN_TCB_PVT_VER_3:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.taskID;
                break;
            case FLCN_TCB_PVT_VER_4:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.taskID;
                break;
            case FLCN_TCB_PVT_VER_5:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
                break;
            case FLCN_TCB_PVT_VER_6:
                taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.taskID;
                break;
            default:
                return NULL;
        }
        return thisFlcn->pFCIF->flcnGetTasknameFromId(taskId);
    }
    else if (pTcb->tcbVer == FLCN_TCB_VER_5)
    {
        FLCN_TCB_PVT *pFlcnTcbPriv = NULL;
        LwU8 taskId;

        //
        // SafeRTOS does not have the task ID in the xTCB. Instead, use the
        // task ID stored in the private TCB.
        //
        flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb5.pvTcbPvt, 1);
        taskId = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
        return thisFlcn->pFCIF->flcnGetTasknameFromId(taskId);
    }
    else
    {
        return NULL;
    }
}

/*!
 * @brief Attempt to get the address of the current TCB. thisFlcn
 * object should be properly set by the caller to point to the
 * target falcon with RTOS loaded.
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
flcnRtosTcbGetLwrrent
(
    FLCN_RTOS_TCB   *pTcb,
    LwU32            port
)
{
    FLCN_SYM    *pMatches;
    LwBool      bExactFound;
    LwU32       count;
    LwU32       symValue;
    LwU32       engineBase = 0;

    if (!thisFlcn || !thisFlcn->pFCIF || !thisFlcn->pFEIF) {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return FALSE;
    }
    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (!flcnSymCheckAutoLoad())
    {
        dprintf("lw: %s please load the symbols first\n", __FUNCTION__);
        return FALSE;
    }

    pMatches = flcnSymFind("pxLwrrentTCB", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        if (FLCN_RTOS_DEREF_DMEM_PTR(engineBase, pMatches->addr, port, &symValue))
        {
            return flcnRtosTcbGet(symValue, port, pTcb);
        }
    }
    return FALSE;
}

/*!
 * Get the corresponding name of the overlay given the overlay id.
 *
 * @param[in]  id        The overlay id
 * @param[in]  symbols   The overlay id names we have search thorugh flcnsym.
 * @param[in]  count     Total count of overlay names.
 *
 * @return     pointer to the string of overlay name
        Overlay ID is valid
   @return     "!error!"
        Overlay ID could not be found
 */
static char *
_searchOverlapNameInFlcnSym
(
    LwU8      id,
    FLCN_SYM *pSymbols,
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
 * The contents of the given TCB, the stack pointed to by the TCB, and
 * attempt to callwlate the maximum stack depth that the task associated
 * with the TCB ever realized. thisFlcn object should be properly set by
 * the caller to point to the target falcon with RTOS loaded.
 *
 * @param pTcb[in] Pointer to the TCB to dump
 * @param port[in] Port to use when reading the stack from DMEM
 * @param size[in] Width of each element in the stack dump
 */
void
flcnRtosTcbDump
(
    FLCN_RTOS_TCB   *pTcb,
    BOOL            bBrief,
    LwU32           port,
    LwU8            size
)
{
    extern  POBJFLCN thisFlcn;
    LwU32           *pStack = NULL;
    LwU32            stackSize = 0;
    LwU8            *pStack8;
    LwU32            maxDepth;
    LwU32            stackSizeBytes;
    LwU32            i;
    LwBool           bExactFound;
    FLCN_SYM         *pMatches;
    LwU32            count;
    char             buffer[24];
    FLCN_TCB_PVT    *pFlcnTcbPriv = NULL;
    LwU32            stackTop = 0;
    LwU8             taskId = 0xFF;
    LwF32            stackUsage;
    LwF32            lwrStackUsg = 0.0;
    LwU32            engineBase = 0;
    LwU32            wordsRead;
    const FLCN_CORE_IFACES *pFCIF = thisFlcn->pFCIF;

    // Validate the tcb, print out error msg if it is corrupted
    if (flcnTcbValidate(pTcb) != 0)
    {
        dprintf("The tcb structure is corrupted, possibly due to stack overflow.\n");
        dprintf("Please check if stack size of the task is large enough.\n");
        return;
    }

    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (pTcb->tcbVer == FLCN_TCB_VER_0)
    {
        stackSize = pTcb->flcnTcb.flcnTcb0.stackDepth;
        stackTop = (LwU32)pTcb->flcnTcb.flcnTcb0.pStack;
        taskId = (LwU8)pTcb->flcnTcb.flcnTcb0.tcbNumber;
    }
    else
    {
        if (pTcb->tcbVer == FLCN_TCB_VER_1)
        {
            flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb1.pvTcbPvt, port);
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_2)
        {
            flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb2.pvTcbPvt, port);
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_3)
        {
            flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb3.pvTcbPvt, port);
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_4)
        {
            flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb4.pvTcbPvt, port);
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_5)
        {
            flcnTcbGetPriv(&pFlcnTcbPriv, (LwU32)pTcb->flcnTcb.flcnTcb5.pvTcbPvt, port);
        }

        // grab flcn pvt tcb specific variables
        switch (pFlcnTcbPriv->tcbPvtVer)
        {
            case FLCN_TCB_PVT_VER_0:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.taskID;
                break;

            case FLCN_TCB_PVT_VER_1:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.taskID;
                break;

            case FLCN_TCB_PVT_VER_2:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.taskID;
                break;

            case FLCN_TCB_PVT_VER_3:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.taskID;
                break;

            case FLCN_TCB_PVT_VER_4:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.taskID;
                break;

            case FLCN_TCB_PVT_VER_5:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
                break;

            case FLCN_TCB_PVT_VER_6:
                stackSize = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.stackSize;
                stackTop  = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.pStack;
                taskId    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.taskID;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (pTcb->tcbVer == FLCN_TCB_VER_3)
        {
            stackTop = pTcb->flcnTcb.flcnTcb3.pcStackBaseAddress;
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_4)
        {
            stackTop = pTcb->flcnTcb.flcnTcb4.pcStackBaseAddress;
        }
        if (pTcb->tcbVer == FLCN_TCB_VER_5)
        {
            stackTop = pTcb->flcnTcb.flcnTcb5.pcStackBaseAddress;
        }
    }

    // read the context of the stack
    pStack = (LwU32 *)malloc(stackSize * sizeof(LwU32));
    if (pStack == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        return;
    }

    stackSizeBytes = stackSize << 2;
    maxDepth = stackSizeBytes;

    wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              stackTop,
                                              LW_TRUE,
                                              stackSize,
                                              port,
                                              pStack);

    if (wordsRead != stackSize)
    {
        dprintf("lw: %s: unable to read stack at address 0x%x. May not be paged in\n",
                __FUNCTION__, stackTop + (wordsRead << 2));
        stackUsage = -1.0;
        lwrStackUsg = -1.0;
    }
    else
    {
        pStack8        = (LwU8*)pStack;

        for (i = 0; i < stackSizeBytes; i++)
        {
            if (pStack8[i] != 0xa5)
                break;
            maxDepth--;
        }

        stackUsage = (float)maxDepth / (float)(stackSize << 2);
    }

    sprintf(buffer, "task%d_", taskId);
    pMatches = flcnSymFind(buffer, FALSE, &bExactFound, &count);

    dprintf("lw:   TCB Address     = 0x%x\n", pTcb->tcbAddr);
    if (pTcb->tcbVer == FLCN_TCB_VER_0)
    {
        dprintf("lw:   Task ID         = 0x%x (%s)\n" ,  pTcb->flcnTcb.flcnTcb0.tcbNumber, pTcb->flcnTcb.flcnTcb0.taskName);
        dprintf("lw:   Priority        = 0x%x\n" , pTcb->flcnTcb.flcnTcb0.priority);
        dprintf("lw:   pStack          = 0x%x\n" ,  pTcb->flcnTcb.flcnTcb0.pStack);
        dprintf("lw:   stackTop        = 0x%x\n", pTcb->flcnTcb.flcnTcb0.pTopOfStack);
        dprintf("lw:   Stack Size      = %d bytes\n" , (int)(pTcb->flcnTcb.flcnTcb0.stackDepth * sizeof(LwU32)));

        lwrStackUsg = (LwF32)(pTcb->flcnTcb.flcnTcb0.pStack + (stackSize << 2) - pTcb->flcnTcb.flcnTcb0.pTopOfStack) /
                      (LwF32)(stackSize << 2);
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

        // grab flcn pvt tcb specific variables
        switch (pFlcnTcbPriv->tcbPvtVer)
        {
            case FLCN_TCB_PVT_VER_0:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_1:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_2:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_3:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_4:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_5:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.usedHeap;
                break;

            case FLCN_TCB_PVT_VER_6:
                taskID         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.taskID;
                privilegeLevel = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.privilegeLevel;
                pStack         = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.pStack;
                stackSize      = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.stackSize;
                usedHeap       = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.usedHeap;
                stackCanary    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.stackCanary;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (pTcb->tcbVer == FLCN_TCB_VER_1)
        {
            priority = pTcb->flcnTcb.flcnTcb1.uxPriority;
            topOfStack = pTcb->flcnTcb.flcnTcb1.pxTopOfStack;
            prvTcbAddr = pTcb->flcnTcb.flcnTcb1.pvTcbPvt;
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_2)
        {
            priority = pTcb->flcnTcb.flcnTcb2.ucPriority;
            topOfStack = pTcb->flcnTcb.flcnTcb2.pxTopOfStack;
            prvTcbAddr = pTcb->flcnTcb.flcnTcb2.pvTcbPvt;
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_3)
        {
            priority = pTcb->flcnTcb.flcnTcb3.ucPriority;
            topOfStack = pTcb->flcnTcb.flcnTcb3.pxTopOfStack;
            prvTcbAddr = pTcb->flcnTcb.flcnTcb3.pvTcbPvt;
            pStack = pTcb->flcnTcb.flcnTcb3.pcStackBaseAddress;
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_4)
        {
            priority = pTcb->flcnTcb.flcnTcb4.uxPriority;
            topOfStack = pTcb->flcnTcb.flcnTcb4.pxTopOfStack;
            prvTcbAddr = pTcb->flcnTcb.flcnTcb4.pvTcbPvt;
            pStack = pTcb->flcnTcb.flcnTcb4.pcStackBaseAddress;
        }
        else if (pTcb->tcbVer == FLCN_TCB_VER_5)
        {
            priority = pTcb->flcnTcb.flcnTcb5.ucPriority;
            topOfStack = pTcb->flcnTcb.flcnTcb5.pxTopOfStack;
            prvTcbAddr = pTcb->flcnTcb.flcnTcb5.pvTcbPvt;
            pStack = pTcb->flcnTcb.flcnTcb5.pcStackBaseAddress;
        }

        dprintf("lw:   Task ID         = 0x%x (%s)\n" , taskID, pFCIF->flcnGetTasknameFromId(taskID));
        dprintf("lw:   Priority        = 0x%x\n" , priority);
        dprintf("lw:   Privilege       = 0x%x\n" , privilegeLevel);
        dprintf("lw:   pStack          = 0x%x\n" , pStack);
        dprintf("lw:   Stack Top       = 0x%x\n" , topOfStack);
        dprintf("lw:   Stack Size      = %d bytes\n", (int)(stackSize * sizeof(LwU32)));
        dprintf("lw:   Used Heap       = %d\n", usedHeap);
        dprintf("lw:   PVT TCB Addr    = 0x%x\n", prvTcbAddr);
        dprintf("lw:   TCB Version     = %d\n", pTcb->tcbVer);
        dprintf("lw:   PVT TCB Version = %d\n", pFlcnTcbPriv->tcbPvtVer);
        if (pFlcnTcbPriv->tcbPvtVer >= FLCN_TCB_PVT_VER_6)
        {
            dprintf("lw:   Stack Canary    = 0x%08x\n", stackCanary);
        }

        lwrStackUsg = (LwF32)(pStack + (stackSize << 2) - topOfStack) / (LwF32)(stackSize << 2);
    }

    dprintf("lw:   Stack Depth     = %d bytes\n" , maxDepth);
    if (stackUsage >= 0)
    {
        dprintf("lw:   Max Stack Usage = %.2f%%\n", stackUsage * 100.0);
        dprintf("lw:   Lwr Stack Usage = %.2f%%\n", lwrStackUsg * 100.0);

        if (stackUsage >= 1.0)
            dprintf("ERROR:   STACK OVERFLOW DETECTED!!!\n");
        else if (stackUsage >= 0.95)
            dprintf("WARNING: The stack is nearly full, consider expanding the stack size.\n");
    }
    else
    {
        dprintf("lw:   Stack is not paged in. Unable to compute stack usage.\n");
    }

    // Print out overlay information if we are using new tcb
    if (pTcb->tcbVer != FLCN_TCB_VER_0)
    {
        int ovlCntImem = 0, ovlCntDmem = 0;
        LwU8 *ovlList = NULL;

        switch (pFlcnTcbPriv->tcbPvtVer)
        {
            case FLCN_TCB_PVT_VER_0:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.ovlCnt;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt0.ovlList;
                break;

            case FLCN_TCB_PVT_VER_1:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.ovlCntImem;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt1.ovlList;
                break;

            case FLCN_TCB_PVT_VER_2:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.ovlCntImem;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt2.ovlList;
                break;

            case FLCN_TCB_PVT_VER_3:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntImemLS +
                             pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntImemHS;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt3.ovlList;
                break;

            case FLCN_TCB_PVT_VER_4:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntImemLS +
                             pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntImemHS;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt4.ovlList;
                break;

            case FLCN_TCB_PVT_VER_5:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntImemLS +
                             pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntImemHS;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt5.ovlList;
                break;

            case FLCN_TCB_PVT_VER_6:
                ovlCntImem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntImemLS +
                             pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntImemHS;
                ovlCntDmem = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.ovlCntDmem;
                ovlList    = pFlcnTcbPriv->flcnTcbPvt.flcnTcbPvt6.ovlList;
                break;

            default:
                dprintf("Cannot retrieve the current pvt tcb or pvt tcb is corrupted");
                break;
        }

        if (ovlCntImem > 0)
        {
            int index;
            FLCN_SYM*   pMatches;
            const char* patterlw0 = "_overlay_id_";
            const char* patterlw1 = "_overlay_id_imem_";
            LwBool      bExactFound;
            LwU32       count;

            // The additional +1 is for the additional _ at the begining
            LwU32       offset = (LwU32)strlen(patterlw1) + 1;

            dprintf("lw:   Attached Imem Overlays (max %d):\n", ovlCntImem);
            pMatches = flcnSymFind(patterlw1, FALSE, &bExactFound, &count);

            // if there are no matched symbols for _overlay_id_imem_
            // then check for legacy _overlay_id_ symbols
            if (pMatches == 0)
            {
                // The additional +1 is for the additional _ at the begining
                offset = (LwU32)strlen(patterlw0) + 1;
                pMatches = flcnSymFind(patterlw0, FALSE, &bExactFound, &count);
            }

            for (index = 0; index < ovlCntImem; ++index)
            {
                LwU8 overlayId = ovlList[index];
                char* overlayName = _searchOverlapNameInFlcnSym(overlayId, pMatches, count);

                if (overlayName == NULL)
                    dprintf("Unable to find corresponding overlap name for id %x.\n", overlayId);
                else if (overlayId != 0)
                    dprintf("lw:                 0x%02x (%s)\n", overlayId, &overlayName[offset]);
            }

        }

        if (ovlCntDmem > 0)
        {
            int index;
            FLCN_SYM*   pMatches;
            const char* pattern = "_overlay_id_dmem_";
            LwBool      bExactFound;
            LwU32       count;

            // The additional +1 is for the additional _ at the begining
            LwU32       offset = (LwU32)strlen(pattern) + 1;

            dprintf("lw:   Attached Dmem Overlays (max %d):\n", ovlCntDmem);
            pMatches = flcnSymFind(pattern, FALSE, &bExactFound, &count);

            for (index = 0; index < ovlCntDmem; ++index)
            {
                LwU8 overlayId = ovlList[ovlCntImem + index];
                char* overlayName = _searchOverlapNameInFlcnSym(overlayId, pMatches, count);

                if (overlayName == NULL)
                    dprintf("Unable to find corresponding overlap name for id %x.\n", overlayId);
                else if (overlayId != 0)
                    dprintf("lw:                 0x%02x (%s)\n", overlayId, &overlayName[offset]);
            }

        }
    }

    if (pFlcnTcbPriv != NULL)
        free(pFlcnTcbPriv);

    // Cleanup
    free(pStack);
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
_flcnFetchTcbInList
(
    const char       *pSymbol,
    LwU32             nItem,
    LwU32             port,
    FLCN_RTOS_TCB   **ppTcbFetched,
    LwU32            *pTotal
)
{
    FLCN_RTOS_TCB    **ppTcbs;
    FLCN_RTOS_TCB     *pTcbs;
    LwU32             *pNTcb;
    FLCN_SYM          *pMatches;
    LwU32              count;
    LwU32              addr;
    LwBool             bExactMatch;
    FLCN_RTOS_XLIST    xList;
    LwU32              idx;
    LwU32              total;
    LwU32              i;
    LwBool             bStatus;
    LwU32              engineBase;

    bStatus = LW_TRUE;
    pMatches = flcnSymFind(pSymbol, LW_FALSE, &bExactMatch, &count);

    if (!bExactMatch)
    {
        dprintf("ERROR: Failed to fetch symbol %s, please verify LwWatch and the symbol being loaded\n", pSymbol);
        return LW_FALSE;
    }

    ppTcbs = (FLCN_RTOS_TCB **)malloc(sizeof(FLCN_RTOS_TCB *) * nItem);
    if (ppTcbs == NULL)
    {
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        return LW_FALSE;
    }
    memset(ppTcbs, 0, sizeof(FLCN_RTOS_TCB *) * nItem);

    pNTcb = (LwU32 *)malloc(sizeof(LwU32) * nItem);
    if (pNTcb == NULL)
    {
        free(ppTcbs);
        dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
        return LW_FALSE;
    }

    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();
    total = 0;
    addr = pMatches->addr;
    for (idx = 0; idx < nItem; ++idx)
    {
        LwU32 listAddr;
        LwU32 wordsRead;

        wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                                  addr,
                                                  LW_TRUE,
                                                  sizeof(FLCN_RTOS_XLIST) >> 2,
                                                  port,
                                                  (LwU32 *)&xList);

        if (wordsRead != (sizeof(FLCN_RTOS_XLIST) >> 2))
        {
            dprintf("ERROR: unable to read TCB data at address 0x%x\n",
                    addr);
            break;
        }

        pNTcb[idx] = xList.numItems;
        total += xList.numItems;
        ppTcbs[idx] = (FLCN_RTOS_TCB *)malloc(sizeof(FLCN_RTOS_TCB) * xList.numItems);
        if (ppTcbs[idx] == NULL)
        {
            dprintf("ERROR: unable to create temporary buffer in function %s\n", __FUNCTION__);
            bStatus = LW_FALSE;
            break;
        }

        listAddr = xList.listEnd.prev;
        for (i = 0; i < xList.numItems; ++i)
        {
            FLCN_RTOS_XLIST_ITEM  item;
            FLCN_RTOS_TCB        *pTcb = &ppTcbs[idx][i];

            wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                                      listAddr,
                                                      LW_TRUE,
                                                      sizeof(item) >> 2,
                                                      port,
                                                      (LwU32 *)&item);
            if (wordsRead != (sizeof(item) >> 2))
            {
                dprintf("ERROR: unable to read item list at 0x%x\n",
                        listAddr);
                break;
            }

            flcnRtosTcbGet(item.owner, port, pTcb);
            listAddr = item.prev;
        }

        addr += sizeof(FLCN_RTOS_XLIST);
    }

    pTcbs = (FLCN_RTOS_TCB *)malloc(sizeof(FLCN_RTOS_TCB) * total);
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
                memcpy(&pTcbs[i++], &ppTcbs[idx][j], sizeof(FLCN_RTOS_TCB));
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
flcnTcbFetchAll
(
    LwU32           port,
    FLCN_RTOS_TCB **ppTcb,
    LwU32          *pLen
)
{
    struct
    {
        char             *pSymbol;
        FLCN_RTOS_TCB    *pTcb;
        LwU32             numTcb;
        LwU32             nItem;
    } tcbLists[] = {
                        {NULL, NULL, 0, configMAX_PRIORITIES},
                        {"xDelayedTaskList1",  NULL, 0, 1},
                        {"xDelayedTaskList2",  NULL, 0, 1},
                        {"xPendingReadyList",  NULL, 0, 1},
                        {"xSuspendedTaskList", NULL, 0, 1}
                    };
    LwU32          idx;
    LwU32          jdx;
    LwU32          nItem = sizeof(tcbLists) / sizeof(tcbLists[1]);
    LwU32          total = 0;
    FLCN_RTOS_TCB *pTcb;
    LwBool         bStatus;
    LwBool         bRes  = LW_TRUE;
    FLCN_SYM      *pMatches;
    LwBool         bExactFound;
    LwU32          count;

    pMatches = flcnSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        LwU64 lwrtosVerNum;
        LwU16 rtosVer;
        LwU32 engineBase = 0;

        engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();

        if (!FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, pMatches->addr, port, &lwrtosVerNum))
        {
            dprintf("ERROR: Failed to fetch lwrtosVersion variable. \n");
            return LW_FALSE;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

        if (rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
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
        bStatus = _flcnFetchTcbInList(tcbLists[idx].pSymbol,
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
    pTcb = (FLCN_RTOS_TCB *)malloc(sizeof(FLCN_RTOS_TCB) * total);
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
 * Print/dump out information for all OsTask TCB's. thisFlcn
 * object should be properly set by the caller to point to the
 * target falcon with RTOS loaded.
 *
 * @param[in]  bBrief  'TRUE' to print a brief message for each TCB; 'FALSE'
 *                     for detailed information.
 */
void
flcnRtosTcbDumpAll
(
    BOOL bBrief
)
{
    FLCN_RTOS_TCB  *pTcb;
    LwU32           len;
    LwU32           i           = 0x0;
    LwU32           engineBase  = 0x0;
    LwU8            width       = 0x04;
    LwBool          bFetched;
    LwU32           port        = 0x1;

    if (!thisFlcn || !thisFlcn->pFCIF || !thisFlcn->pFEIF) {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return;
    }
    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (!flcnSymCheckAutoLoad())
    {
        dprintf("lw: %s please load the symbols first\n", __FUNCTION__);
        return;
    }

    bFetched = flcnTcbFetchAll(port, &pTcb, &len);
    if (!bFetched)
    {
        dprintf("ERROR: Failed to load tcbs from RTOS, please check LwWatch.\n");
        return;
    }

    for (i = 0; i < len; i++)
    {
        dprintf("lw:========================================\n");
        flcnRtosTcbDump(&pTcb[i], bBrief, 1, width);
        dprintf("lw:\n");
    }

    if (bBrief)
    {
        dprintf("lw:\n");
        dprintf("lw:\tHSM = High-Stack Mark\n");
    }
    return;
}


/*!
 * Print out information for all OsTask Dmem Overlays. thisFlcn
 * object should be properly set by the caller to point to the
 * target falcon with RTOS loaded.
 */
void
flcnRtosDmemOvlDumpAll()
{
    FLCN_RTOS_TCB  *pTcb             = NULL;
    FLCN_TCB_PVT   *pFlcnTcbPriv     = NULL;
    LwU32           len              = 0;
    LwU32           i                = 0;
    LwU32           engineBase       = 0;
    LwU32           port             = 1;
    LwBool          bFetched         = LW_FALSE;
    FLCN_SYM       *pMatches         = NULL;
    const char     *pOverlayCountStr = "_overlay_id_dmem__count";
    LwBool          bExactFound      = LW_FALSE;
    LwU32           count            = 0;

    if ((thisFlcn == NULL) || (thisFlcn->pFCIF == NULL) ||
        (thisFlcn->pFEIF == NULL))
    {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return;
    }

    engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (!flcnSymCheckAutoLoad())
    {
        dprintf("lw: %s please load symbols first\n", __FUNCTION__);
        return;
    }

    bFetched = flcnTcbFetchAll(port, &pTcb, &len);
    if (!bFetched)
    {
        dprintf("ERROR: Failed to load tcbs from RTOS, please check LwWatch.\n");
        return;
    }

    // Validate the tcb, print out error msg if it is corrupted
    if (flcnTcbValidate(&(pTcb[0])) != 0)
    {
        dprintf("TCB %d: structure is corrupted, possibly due to stack overflow.\n", i);
        dprintf("Please check if stack size of the task is large enough.\n");
        return;
    }

    if (pTcb[0].tcbVer == FLCN_TCB_VER_1)
    {
        if (!flcnTcbGetPriv(&pFlcnTcbPriv,
                            (LwU32)pTcb[0].flcnTcb.flcnTcb1.pvTcbPvt, port))
        {
            dprintf("ERROR: Failed to load PVT TCB\n");
            return;
        }
    }
    else if (pTcb[0].tcbVer == FLCN_TCB_VER_2)
    {
        if (!flcnTcbGetPriv(&pFlcnTcbPriv,
                            (LwU32)pTcb[0].flcnTcb.flcnTcb2.pvTcbPvt, port))
        {
            dprintf("ERROR: Failed to load PVT TCB\n");
            return;
        }
    }
    else if (pTcb[0].tcbVer == FLCN_TCB_VER_3)
    {
        if (!flcnTcbGetPriv(&pFlcnTcbPriv,
                            (LwU32)pTcb[0].flcnTcb.flcnTcb3.pvTcbPvt, port))
        {
            dprintf("ERROR: Failed to load PVT TCB\n");
            return;
        }
    }
    else if (pTcb[0].tcbVer == FLCN_TCB_VER_4)
    {
        if (!flcnTcbGetPriv(&pFlcnTcbPriv,
                            (LwU32)pTcb[0].flcnTcb.flcnTcb4.pvTcbPvt, port))
        {
            dprintf("ERROR: Failed to load PVT TCB\n");
            return;
        }
    }
    else if (pTcb[0].tcbVer == FLCN_TCB_VER_5)
    {
        if (!flcnTcbGetPriv(&pFlcnTcbPriv,
                            (LwU32)pTcb[0].flcnTcb.flcnTcb5.pvTcbPvt, port))
        {
            dprintf("ERROR: Failed to load PVT TCB\n");
            return;
        }
    }
    else
    {
        dprintf("TCB Version %d not supported!\n", pTcb[0].tcbVer);
        return;
    }

    pMatches = flcnSymFind(pOverlayCountStr, FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        const char *pOvlIDDmemStr   = "_overlay_id_dmem";
        const char *pOvlStartAddr   = "_dmem_ovl_start_address";
        const char *pOvlSizeLwrr    = "_dmem_ovl_size_lwrrent";
        const char *pOvlSizeMax     = "_dmem_ovl_size_max";
        LwU32       dmOvlCount      = 0;
        LwU32       startAddr       = 0;
        LwU32       sizeLwrr        = 0;
        LwU32       sizeMax         = 0;

        dmOvlCount = pMatches->addr;

        // Get array addresses
        pMatches = flcnSymFind(pOvlStartAddr, FALSE, &bExactFound, &count);
        if (!bExactFound)
        {
            dprintf("ERROR: DMEM Overlay Start Address array not found!\n");
            return;
        }
        startAddr = pMatches->addr;

        pMatches = flcnSymFind(pOvlSizeLwrr, FALSE, &bExactFound, &count);
        if (!bExactFound)
        {
            dprintf("ERROR: DMEM Overlay Size Current array not found!\n");
            return;
        }
        sizeLwrr = pMatches->addr;

        pMatches = flcnSymFind(pOvlSizeMax, FALSE, &bExactFound, &count);
        if (!bExactFound)
        {
            dprintf("ERROR: DMEM Overlay Size Max array not found!\n");
            return;
        }
        sizeMax = pMatches->addr;

        // read in the arrays
        {
            LwU32  *pStartAddrs         = NULL;
            LwU16  *pSizesLwrrHandle    = NULL;
            LwU16  *pSizesLwrr          = NULL;
            LwU16  *pSizesMaxHandle     = NULL;
            LwU16  *pSizesMax           = NULL;
            LwU32   wordsRead           = 0;
            LwU32   i                   = 0;

            pStartAddrs = (LwU32 *)malloc(dmOvlCount * sizeof(LwU32));
            if (pStartAddrs == NULL)
            {
                dprintf("ERROR: Failed to malloc space for Start Addr array!\n");
                return;
            }

            wordsRead =
                thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              startAddr,
                                              LW_TRUE,
                                              dmOvlCount,
                                              1,
                                              (LwU32 *)pStartAddrs);

            if (wordsRead != dmOvlCount)
            {
                dprintf("ERROR: Start Addr array: words read does not match Overlay Count!  Words read: %d\n",
                        wordsRead);
                goto FreeAndExit;
            }

            pSizesLwrrHandle = pSizesLwrr = (LwU16 *)malloc(dmOvlCount * sizeof(LwU32));
            if (pSizesLwrrHandle == NULL)
            {
                dprintf("ERROR: Failed to malloc space for Current Sizes array!\n");
                goto FreeAndExit;
            }

            wordsRead =
                thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              sizeLwrr,
                                              LW_TRUE,
                                              dmOvlCount,
                                              1,
                                              (LwU32 *)pSizesLwrr);

            if (wordsRead != dmOvlCount)
            {
                dprintf("ERROR: Current Sizes array: words read does not match Overlay Count!  Words read: %d\n",
                        wordsRead);
                goto FreeAndExit;
            }
            // Adjust pointer to compensate for dword alignment in flcnDmemRead
            if (sizeLwrr % 4)
                pSizesLwrr++;

            pSizesMaxHandle = pSizesMax = (LwU16 *)malloc(dmOvlCount * sizeof(LwU32));
            if (pSizesMaxHandle == NULL)
            {
                dprintf("ERROR: Failed to malloc space for Max Sizes array!\n");
                goto FreeAndExit;
            }

            wordsRead =
                thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              sizeMax,
                                              LW_TRUE,
                                              dmOvlCount,
                                              1,
                                              (LwU32 *)pSizesMax);

            if (wordsRead != dmOvlCount)
            {
                dprintf("ERROR: Max Sizes array: words read does not match Overlay Count!  Words read: %d\n",
                        wordsRead);
                goto FreeAndExit;
            }
            // Adjust pointer to compensate for dword alignment in flcnDmemRead
            if (sizeMax % 4)
                pSizesMax++;

            // print results
            pMatches = flcnSymFind(pOvlIDDmemStr, FALSE, &bExactFound, &count);
            if (count > 0)
            {
                FLCN_SYM   *pLSMatch = NULL;

                for (i = 0; i < dmOvlCount;  i++)
                {
                    pLSMatch = pMatches;
                    while (pLSMatch)
                    {
                        if (pLSMatch->addr == i)
                        {
                            break;
                        }
                        pLSMatch = pLSMatch->pTemp;
                    }
                    dprintf("    id = 0x%02x    start = 0x%04x    lwrSize = 0x%04x    maxSize = 0x%04x    // %s\n",
                            i, pStartAddrs[i], pSizesLwrr[i], pSizesMax[i],
                            (NULL != pLSMatch) ? pLSMatch->name : "Name not found");
                }
            }
            else
            {
                dprintf("ERROR %s not found!\n", pOvlIDDmemStr);
            }

        FreeAndExit:;
            if (NULL != pStartAddrs)
                free(pStartAddrs);
            if (NULL != pSizesLwrrHandle)
                free(pSizesLwrrHandle);
            if (NULL != pSizesMaxHandle)
                free(pSizesMaxHandle);
            return;
        }
    }
    else
    {
        dprintf("ERROR:  DMEM Overlay Count not found!  Check symbols.\n");
        return;
    }

    dprintf("\n");
}

/*!
 *
 * Dump all RTOS scheduler information (ready-lists, suspended list, etc ...)
 * thisFlcn object should be properly set by the caller to point to the
 * target falcon with RTOS loaded.
 *
 * @param[in]   bTable  Format of dumping
 */

void
flcnRtosSchedDump
(
    BOOL   bTable
)
{
    FLCN_SYM   *pMatches;
    const char *pReadyTaskListName;
    LwBool     bExactFound;
    LwU32      count;
    LwS32      i;
    LwU32      value;
    LwU32      engineBase;

    char *pSimpleLists[] =
    {
        "xSuspendedTaskList"  ,
        "xDelayedTaskList1"   ,
        "xDelayedTaskList2"   ,
    };

    char *pSchedulerVars[] =
    {
        "xSchedulerRunning"     ,
        "uxSchedulerSuspended"  ,
        "uxDmaSuspend"          ,
        "uxLwrrentNumberOfTasks",
        "xTickCount"            ,
        "uxMissedTicks"         ,
    };

    if (!thisFlcn || !thisFlcn->pFCIF || !thisFlcn->pFEIF) {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return;
    }
    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (!flcnSymCheckAutoLoad())
    {
        dprintf("lw: %s please load the symbols first\n", __FUNCTION__);
        return;
    }

    pMatches = flcnSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
    if (bExactFound)
    {
        LwU64 lwrtosVerNum;
        LwU16 rtosVer;

        if (!FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, pMatches->addr, 1, &lwrtosVerNum))
        {
            dprintf("ERROR: Failed to fetch lwrtosVersion variable. \n");
            return;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

        if (rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
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

    pMatches = flcnSymFind(pReadyTaskListName, FALSE, &bExactFound, &count);
    if (!bExactFound)
    {
        dprintf("lw: Cannot exact find the ready task list symbol. \n");
        return;
    }
    count = pMatches->size / sizeof(FLCN_RTOS_XLIST);

    //
    // The ReadyTaskLists is an array of task list with index 0 being
    // the lowest priority task list
    //
    for (i = count - 1; i >= 0; i--)
    {
        char name[32] = {'\0'};
        sprintf(name, "_%s[%d]", pReadyTaskListName, i);

        FLCN_RTOS_PRINT_SEPARATOR();
        _XListPrint(pMatches->addr + (i*sizeof(FLCN_RTOS_XLIST)), name, bTable);
        dprintf("lw:\n");
    }

    //
    // suspended task list
    //

    for (i = 0; i < (LwS32)(sizeof(pSimpleLists)/sizeof(char*)); i++)
    {
        pMatches = flcnSymFind(pSimpleLists[i], FALSE, &bExactFound, &count);
        if ((!bExactFound) || (count != 1))
        {
            dprintf("lw: Cannot exact find %s\n", pSimpleLists[i]);
            continue;
        }
        FLCN_RTOS_PRINT_SEPARATOR();
        _XListPrint(pMatches->addr, pMatches->name, bTable);
        dprintf("lw:\n");
    }

    FLCN_RTOS_PRINT_SEPARATOR();
    dprintf("lw:\n");
    dprintf("lw: General:\nlw:\n");

    for (i = 0; i < (LwS32)(sizeof(pSchedulerVars)/sizeof(char*)); i++)
    {
        pMatches = flcnSymFind(pSchedulerVars[i], FALSE, &bExactFound, &count);
        if (count == 1)
        {
            if (FLCN_RTOS_DEREF_DMEM_PTR(engineBase, pMatches->addr, 1, &value))
            {
                dprintf("lw: [0x%04x] %24s : 0x%08x\n", pMatches->addr, pMatches->name, value);
            }
        }
    }
    return;
}

/*!
 * @brief  Unload the cache used to print queues
 */
static void
_flcnEvtqUnloadSymbol(void)
{
    if (bEvtqSymbolLoaded)
    {
        free(flcnEvtqInfo.ppEvtqSyms);
        free(flcnEvtqInfo.pMem);
        bEvtqSymbolLoaded = LW_FALSE;
    }
}

/*!
 * @brief  Load all symbols in section B into struct flcnEvtqInfo
 *
 *
 * @return TRUE
 *      Symbols loaded correctly
 * @return FALSE
 *      Unable to load symbols
 */
static BOOL
_flcnEvtqLoadSymbol(void)
{
    const char  *pcQueueHeader;
    const char  *pcHeap            = "_heap";
    const char  *pcHeapEnd         = "_heap_end";
    FLCN_SYM    *pMatches;
    FLCN_SYM    *pSymIter;
    LwBool       bExactFound;
    LwU32        count;
    LwU32        i;
    LwU32        engineBase    = 0x0;
    LwU32        wordsRead;

    if (!thisFlcn || !thisFlcn->pFCIF || !thisFlcn->pFEIF) {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return FALSE;
    }

    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();
    if (!bEvtqSymbolLoaded)
    {
        if (!flcnSymCheckAutoLoad())
        {
            dprintf("Unable to load .nm file, without which the flcnevtq cannot work, please load .nm with \"flcnsym -l [.nmFile]\" manually\n");
            return FALSE;
        }

        // Check for queue list head symbol
        pMatches = flcnSymFind("LwosUcodeVersion", FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            LwU64 lwrtosVerNum;
            LwU16 rtosVer;

            if (!FLCN_RTOS_DEREF_DMEM_PTR_64(engineBase, pMatches->addr, 1, &lwrtosVerNum))
            {
                dprintf("ERROR: Failed to fetch LwosUcodeVersion variable. \n");
                return FALSE;
            }
            rtosVer = DRF_VAL64(_RM, _RTOS_VERSION, _RTOS_TYPE, lwrtosVerNum);

            if (rtosVer == LW_RM_RTOS_VERSION_RTOS_TYPE_SAFERTOS)
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

        pMatches = flcnSymFind(pcQueueHeader, LW_FALSE, &bExactFound, &count);
        if (count == 0)
        {
            flcnEvtqInfo.qHead = 0;
        }
        else
        {
            flcnEvtqInfo.qHead = pMatches->addr;
        }

        // Check symbol heap and _heap_end
        pMatches = flcnSymFind(pcHeap, LW_FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            flcnEvtqInfo.heap = pMatches->addr;
        }
        else
        {
            flcnEvtqInfo.heap = 0;
        }

        pMatches = flcnSymFind(pcHeapEnd, LW_FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            flcnEvtqInfo.heapEnd = pMatches->addr;
        }
        else
        {
            flcnEvtqInfo.heap = thisFlcn->pFCIF->flcnDmemGetSize(engineBase);
        }

        pMatches = flcnSymFind("_", LW_FALSE, &bExactFound, &count);
        pSymIter = pMatches;
        flcnEvtqInfo.nEvtqSyms = 0;

        while (pSymIter != NULL)
        {
            if ((pSymIter->section != 'T') &&
                (pSymIter->section != 't') &&
                (!pSymIter->bSizeEstimated) &&
                (strstr(pSymIter->name, pcQueueHeader) == NULL)&&
                (pSymIter->addr < flcnEvtqInfo.heapEnd))
            {
                ++flcnEvtqInfo.nEvtqSyms;
            }
            pSymIter = pSymIter->pTemp;
        }

        flcnEvtqInfo.ppEvtqSyms = (FLCN_SYM **)malloc(sizeof(FLCN_SYM *) * flcnEvtqInfo.nEvtqSyms);

        i = 0;
        pSymIter = pMatches;
        flcnEvtqInfo.addrStart = 0xFFFFFFFF;
        flcnEvtqInfo.addrEnd = 0;
        while (pSymIter != NULL)
        {
            if ((pSymIter->section != 'T') &&
                (pSymIter->section != 't') &&
                (!pSymIter->bSizeEstimated) &&
                (strstr(pSymIter->name, pcQueueHeader) == NULL) &&
                (pSymIter->addr < flcnEvtqInfo.heapEnd))
            {
                flcnEvtqInfo.ppEvtqSyms[i] = pSymIter;
                ++i;

                if (pSymIter->addr < flcnEvtqInfo.addrStart)
                    flcnEvtqInfo.addrStart = pSymIter->addr;

                flcnEvtqInfo.addrEnd = pSymIter->addr + pSymIter->size;
            }
            pSymIter = pSymIter->pTemp;
        }

        if (flcnEvtqInfo.addrEnd <= flcnEvtqInfo.addrStart)
        {
            dprintf("ERROR: Invalid information got from .nm file, please validate the correctness of .nm file.\n");
            return FALSE;
        }

        flcnEvtqInfo.memSize = (flcnEvtqInfo.addrEnd - flcnEvtqInfo.addrStart + sizeof(LwU32) - 1) >> 2 << 2;

        flcnEvtqInfo.pMem = (LwU8 *)malloc(flcnEvtqInfo.memSize);
        if (flcnEvtqInfo.pMem == NULL)
        {
            dprintf("ERROR: Failed to malloc space to cache information for flcnevtq!\n");
            return FALSE;
        }

        wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                                  flcnEvtqInfo.addrStart,
                                                  LW_TRUE,
                                                  flcnEvtqInfo.memSize >> 2,
                                                  1,
                                                  (LwU32 *)flcnEvtqInfo.pMem);

        if (wordsRead == (flcnEvtqInfo.memSize >> 2))
        {
            bEvtqSymbolLoaded = LW_TRUE;
        }
    }

    return bEvtqSymbolLoaded;
}

/*!
 * @brief  Expand the buffer to store the queues, lwrrently will just double
 *         the buffer size
 *
 * @param[out]  ppQueue     Stored all the queues that have been fetched
 * @param[out]  pAddrs      Stored all the addresses of the queues
 * @param[out]  ppQNames    Stored all the names of the queue in a char array
 * @param[out]  pArraySize  Size of the current array
 * @param[in]   strSize     Size of each queue name (fixed)
 *
 * @return    LW_TRUE
 *      Successfully expand the buffer
 * @return    LW_FALSE
 *      Failed to expand buffer
 */
static LwBool
_flcnQueueExpBuf
(
    FLCN_RTOS_XQUEUE **ppQueueArray,
    LwU32            **ppAddrs,
    char             **ppQNameBuf,
    LwU32             *pArraySize,
    LwU32              strSize
)
{
        LwU32             newSize = *pArraySize * 2;
        FLCN_RTOS_XQUEUE *pNewQueue;
        LwU32            *pNewAddr;
        char             *pQNameBuf;

        pNewQueue = (FLCN_RTOS_XQUEUE *)malloc(sizeof(FLCN_RTOS_XQUEUE) * newSize);
        if (pNewQueue == NULL)
        {
            dprintf("ERROR: Out of memory!\n");
            return LW_FALSE;
        }

        pNewAddr = (LwU32 *)malloc(sizeof(FLCN_RTOS_XQUEUE) * newSize);
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

        memcpy(pNewQueue, *ppQueueArray, sizeof(FLCN_RTOS_XQUEUE) * (*pArraySize));
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
flcnQueueFetchAll
(
    FLCN_RTOS_XQUEUE **ppQueue,
    LwU32            **ppAddrs,
    char             **ppQNames,
    LwU32             *pStrSize,
    LwU32             *pNum
)
{
    LwU32             idx;
    LwU32             qAddr;
    LwU32             arraySize;
    FLCN_RTOS_XQUEUE  xQueue;
    FLCN_RTOS_XQUEUE *pQueueArray;
    LwU32            *pAddrs;
    char             *pQNameBuf;
    LwU32             strBuffer;
    LwU32             strIdx;
    LwBool            bLoadedBefore;
    LwU32             engineBase = 0x0;
    LwU32             wordsRead;

    //
    // NOTE:
    // These strings are used ONLY as a fallback option for old drivers without
    // next pointer in the queue structure, they are subjected to be moved
    // in around 6 months (April/2015).
    //
    const char *pQueueNames[] = {
        // Queue names for PMU
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
        "_seqRunSemaphore",

        // Queue names for DPU
        "DpuMgmtCmdQueueMutex",
        "DpuMgmtMsgQueueMutex",
        "DpuMgmtCmdDispQueue",
        "RegCacheQueue",

        // Queue names for SEC2
        "Sec2CmdMgmtCmdDispQueue",
        "Sec2MgmtMsgQueueMutex",
        "Sec2MgmtCmdQueueMutex"
    };

    //
    // If the symbols are loaded when entering this function, then it would not
    // be unloaded during exit, the function who loads the symbol should also
    // be the one who unloads it. So that we can reduce number of times in reading
    // the whole DMEM, which consuming a lot of time.
    //
    bLoadedBefore = bEvtqSymbolLoaded;
    if (!_flcnEvtqLoadSymbol())
    {
        return LW_FALSE;
    }

    arraySize = 32;
    pQueueArray = (FLCN_RTOS_XQUEUE *) malloc(sizeof(FLCN_RTOS_XQUEUE) * arraySize);
    pAddrs = (LwU32 *) malloc(sizeof(LwU32) * arraySize);

    *pStrSize = 64;
    strBuffer = 32 * (*pStrSize);
    pQNameBuf = (char*)malloc(sizeof(char) * strBuffer);

    idx = 0;
    strIdx = 0;
    engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (flcnEvtqInfo.qHead != 0)
    {
        wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                                  flcnEvtqInfo.qHead,
                                                  LW_TRUE,
                                                  1,
                                                  1,
                                                  &qAddr);
        if (wordsRead != 1)
        {
            dprintf("ERROR: Unable to read data at address 0x%x\n",
                    flcnEvtqInfo.qHead);
            return LW_FALSE;
        }

        while (qAddr != 0)
        {
             if (!_flcnSymXQueueGetByAddress(qAddr, &xQueue))
             {
                break;
             }
             pAddrs[idx] = qAddr;
             pQueueArray[idx] = xQueue;
             _flcnGetQueueName(qAddr, &pQNameBuf[idx * (*pStrSize)], *pStrSize);

             qAddr = xQueue.next;
             ++idx;

             if (idx == arraySize)
             {
                if (!_flcnQueueExpBuf(&pQueueArray, &pAddrs, &pQNameBuf, &arraySize, *pStrSize))
                    break;
             }
        }
    }
    else
    {
        // Fallback option for queue implementation without next pointer
        FLCN_SYM *pMatches;
        LwU32     i;
        LwBool    bExactFound;
        LwU32     count;

        for (i = 0; i < sizeof(pQueueNames)/sizeof(char *); i++)
        {
            LwU32 actSize;

            pMatches = flcnSymFind(pQueueNames[i], LW_FALSE, &bExactFound, &count);
            if ((!bExactFound) || (count != 1) || (!FLCN_RTOS_DEREF_DMEM_PTR(engineBase, pMatches->addr, 1, &qAddr)))
                continue;

             if (!_flcnSymXQueueGetByAddress(qAddr, &xQueue))
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
                if (!_flcnQueueExpBuf(&pQueueArray, &pAddrs, &pQNameBuf, &arraySize, *pStrSize))
                    break;
             }
        }
    }

    // If the current function loads the symbol, it would be one that unloads it.
    if (!bLoadedBefore)
    {
        _flcnEvtqUnloadSymbol();
    }
    *pNum = idx;
    *ppQueue = pQueueArray;
    *ppAddrs = pAddrs;
    *ppQNames = pQNameBuf;

    return LW_TRUE;
}

/*!
 * Dump information on all known event queues.
 *
 * @param[in]  bSummary     The queue should be printed in summary view(TRUE),
 *                          or detail view(FALSE)
 */
void
flcnRtosEventQueueDumpAll
(
    BOOL bSummary
)
{
    LwU32             idx;
    LwU32            *pAddrs;
    FLCN_RTOS_XQUEUE *pQueue;
    char             *pQNames;
    LwU32             nQueue;
    LwU32             strSize;

    FLCN_RTOS_PRINT_SEPARATOR();
    dprintf("lw: PMU Event Queues:\n");
    if (bSummary)
        dprintf("lw: Idx Type    Size         Queue Name    ItmSize  MsgWait   head      tail   readfrom   writeto\n");

    if (!flcnQueueFetchAll(&pQueue, &pAddrs, &pQNames, &strSize, &nQueue))
    {
        dprintf("ERROR: Failed to fetch queue informations, please check LwWatch!\n");
        _flcnEvtqUnloadSymbol();
        return;
    }

    for (idx = 0; idx < nQueue; ++idx)
    {
        if (bSummary)
        {
            _flcnXQueuePrint(&pQueue[idx], &pQNames[strSize * idx], pAddrs[idx], idx);
        }
        else
        {
            FLCN_RTOS_PRINT_SEPARATOR();
            _flcnXQueueFullPrint(&pQueue[idx], &pQNames[strSize * idx], pAddrs[idx], idx);
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
flcnRtosEventQueueDumpByAddr
(
    LwU32 qAddr
)
{
    FLCN_RTOS_XQUEUE  queue;

    if (!_flcnEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for flcnevtq, with which the function wouldn't work!\n");
        return;
    }

    if (!_flcnSymXQueueGetByAddress(qAddr, &queue))
    {
        dprintf("ERROR: The address 0x%x is not storing a valid queue structure.\n", qAddr);
    }

    FLCN_RTOS_PRINT_SEPARATOR();
    dprintf("lw: Falcon Event Queue:\n");
    _flcnXQueueFullPrint(&queue, NULL, qAddr, 0);
    _flcnEvtqUnloadSymbol();

    return;
}

/*!
 * Dump information on a specific event queue (identified by task ID).
 *
 * @param[in]  qTaskId     Task ID whose queue we want to print
 */
void
flcnRtosEventQueueDumpByTaskId
(
    LwU32 qTaskId
)
{
    LwU8 queueId = 0U;

    if (!_flcnEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for flcnevtq, with which the function wouldn't work!\n");
        goto flcnRtosEventQueueDumpByTaskId_exit;
    }

    if (!_flcnGetQueueIdFromTaskId(qTaskId, &queueId))
    {
        goto flcnRtosEventQueueDumpByTaskId_exit;
    }

    flcnRtosEventQueueDumpByQueueId((LwU32)queueId);

flcnRtosEventQueueDumpByTaskId_exit:
    _flcnEvtqUnloadSymbol();
}

/*!
 * Dump information on a specific event queue (identified by queue ID).
 *
 * @param[in]  qTaskId     ID of queue we want to print
 */
void
flcnRtosEventQueueDumpByQueueId
(
    LwU32 queueId
)
{
    LwU32 qAddr;
    FLCN_RTOS_XQUEUE  queue;
    LwU32             engineBase = 0;

    if (!_flcnEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for flcnevtq, with which the function wouldn't work!\n");
        goto flcnRtosEventQueueDumpByQueueId_exit;
    }

    if (!_flcnGetQueueAddrFromQueueId((LwU8)queueId, &qAddr))
    {
        goto flcnRtosEventQueueDumpByQueueId_exit;
    }

    // Get the address for the queue.
    engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();
    FLCN_RTOS_DEREF_DMEM_PTR(engineBase, qAddr, 1, &qAddr);

    if (!_flcnSymXQueueGetByAddress(qAddr, &queue))
    {
        dprintf("ERROR: The queue ID 0x%x does not have a matching queue!\n", queueId);
        goto flcnRtosEventQueueDumpByQueueId_exit;
    }

    FLCN_RTOS_PRINT_SEPARATOR();
    dprintf("lw: Falcon Event Queue:\n");
    _flcnXQueueFullPrint(&queue, NULL, qAddr, 0);

flcnRtosEventQueueDumpByQueueId_exit:
    _flcnEvtqUnloadSymbol();
}

/*!
 * Dump information on a specific event queue (identified by symbol).
 * thisFlcn object should be properly set by the caller to point to the
 * target falcon with RTOS loaded.
 *
 * @param[in]  pSym  Name of the symbol for the queue
 */
void
flcnRtosEventQueueDumpBySymbol
(
    const char *pSym
)
{
    FLCN_SYM         *pMatches;
    LwBool            bExactFound;
    LwU32             qAddr;
    LwU32             count;
    FLCN_RTOS_XQUEUE  queue;
    LwU32             engineBase;

    if (!flcnSymCheckAutoLoad())
    {
        dprintf("ERROR: Failed to .nm files, with which the flcnevtq function wouldn't work!\n");
        return;
    }

    if (!_flcnEvtqLoadSymbol())
    {
        dprintf("ERROR: Failed to load symbols for flcnevtq, with which the function wouldn't work!\n");
        return;
    }

    engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();
    pMatches = flcnSymFind(pSym, LW_FALSE, &bExactFound, &count);
    if (count == 0)
    {
        return;
    }
    if (count > 1)
    {
        dprintf("Error: Muliple symbols found with name <%s>\n", pSym);
        return;
    }

    if (pMatches->addr == 0 || !FLCN_RTOS_DEREF_DMEM_PTR(engineBase, pMatches->addr, 1, &qAddr))
    {
        dprintf("Error: queue pointer <0x%04x> is invalid.\n", pMatches->addr);
        return;
    }

    if (qAddr == 0)
    {
        dprintf("Error: queue address <0x%04x> is invalid.\n", qAddr);
        return;
    }

    if (!_flcnSymXQueueGetByAddress(qAddr, &queue))
    {
        dprintf("The address 0x%x is not storing a valid queue structure.\n", qAddr);
        return;
    }

    FLCN_RTOS_PRINT_SEPARATOR();
    dprintf("lw: Falcon Event Queues:\n");
    _flcnXQueueFullPrint(&queue, &pMatches->name[1], qAddr, 0);
    _flcnEvtqUnloadSymbol();
}

/* -------------------------- Helper routines ------------------------------ */

#undef  offsetof
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#undef  XLIST_FMT1
#define XLIST_FMT1        "lw:       [0x%04x]%16s : 0x%08x\n"

#undef  XLIST_FMT2
#define XLIST_FMT2        "lw:       [0x%04x]%16s : 0x%08x    [0x%04x]%7s : 0x%08x\n"

#undef  XLIST_ITEM_ADDR
#define XLIST_ITEM_ADDR(base, item) ((LwU32)((base)+offsetof(FLCN_RTOS_XQUEUE, item)))


/*!
 * Print an OpenRTOS XLIST in human-readible form. thisFlcn object should
 * be set properly by the caller to point to the target falcon object with
 * OpenRTOS loaded.
 *
 * @param[in]  listAddr  DMEM address of the list
 * @param[in]  pName     Name for the list being printed since it not contained
 *                       in the list structure itself.
 * @param[in]  bTable    'TRUE' to print in table-form; 'FALSE' to print in
 *                       list-form.
 */
static void
_XListPrint
(
    LwU32   listAddr,
    char    *pName,
    BOOL    bTable
)
{
    FLCN_RTOS_XLIST         xList;
    FLCN_RTOS_XLIST_ITEM    listItem;
    FLCN_RTOS_TCB           tcb;
    const FLCN_CORE_IFACES *pFCIF = NULL;
    LwU32   i           = 0x0;
    LwU32   addr        = 0x0;
    LwU32   length      = 0x0;
    LwU32   engineBase  = 0x0;
    BOOL    bTcbFound   = FALSE;
    LwU32   wordsRead;

    if (!thisFlcn || !thisFlcn->pFCIF || !thisFlcn->pFEIF) {
        dprintf("lw: %s thisFlcn object is invalid\n", __FUNCTION__);
        return;
    }
    pFCIF       = thisFlcn->pFCIF;
    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    wordsRead = pFCIF->flcnDmemRead(engineBase,
                                    listAddr,
                                    LW_TRUE,
                                    sizeof(FLCN_RTOS_XLIST) >> 2,
                                    1,
                                    (LwU32*)&xList);
    if (wordsRead != (sizeof(FLCN_RTOS_XLIST) >> 2))
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
        length = pFCIF->flcnDmemRead(engineBase,
                                     addr,
                                     LW_TRUE,
                                     sizeof(FLCN_RTOS_XLIST_ITEM) >> 2,
                                     1,
                                     (LwU32*)&listItem);

        if (length != (sizeof(FLCN_RTOS_XLIST_ITEM) >> 2))
        {
            //
            // Most likely the addr went wrong (because the RTOS
            // data structure is not probably initialized or the caller
            // just passed the wrong address)
            //
            break;
        }

        bTcbFound = flcnRtosTcbGet(listItem.owner, 1, &tcb);


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
                    bTcbFound ? flcnRtosGetTasknameFromTcb(&tcb) : "");
        }
        else
        {
            dprintf("lw:\n");
            dprintf("lw: [%d] :[0x%04x]  itemValue : 0x%08x\n", i , addr+0x00, listItem.itemValue);
            dprintf("lw:       [0x%04x]       next : 0x%04x\n"     , addr+0x04, listItem.next);
            dprintf("lw:       [0x%04x]       prev : 0x%04x\n"     , addr+0x08, listItem.prev);
            dprintf("lw:       [0x%04x]      owner : 0x%04x <%s>\n", addr+0x0c, listItem.owner,
                                                                                bTcbFound
                                                                                ? flcnRtosGetTasknameFromTcb(&tcb)
                                                                                : "");
            dprintf("lw:       [0x%04x]  container : 0x%04x\n"     , addr+0x10, listItem.container);
        }

        addr = listItem.next;
    }
    return;
}

/*!
 * @brief  Find the name for the queue
 *
 * @param[in]  qAddr              Address of the queue
 * @param[in]  pcQueueNameBuf     Buffer to hold the queue name
 * @param[in]  bufLen             Size of the buffer
 */
static void
_flcnGetQueueName
(
 LwU32  qAddr,
 char  *pcQueueNameBuf,
 LwU32  bufLen
)
{
    BOOL   bVarUnique = TRUE;
    BOOL   bOffsetUnique = TRUE;
    LwU32  targetIdx = flcnEvtqInfo.nEvtqSyms;
    LwU32  offset = 0;
    char   buf[128];
    char  *pcNameChosen;
    LwU32  i;
    LwU32  j;

    if ((!_flcnEvtqLoadSymbol()) || (qAddr == 0))
    {
        pcNameChosen = "!unknown!";
    }
    else
    {
        for (i = 0; i < flcnEvtqInfo.nEvtqSyms; ++i)
        {
            LwU32 pos = flcnEvtqInfo.ppEvtqSyms[i]->addr - flcnEvtqInfo.addrStart;

            for (j = 0; j < flcnEvtqInfo.ppEvtqSyms[i]->size; j += 4, pos += 4)
            {
                if ((*(LwU32*)&flcnEvtqInfo.pMem[pos]) == qAddr)
                {
                    if ((targetIdx != i) && (targetIdx < flcnEvtqInfo.nEvtqSyms))
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

        if ((bVarUnique) && (targetIdx < flcnEvtqInfo.nEvtqSyms))
        {
            if (flcnEvtqInfo.ppEvtqSyms[targetIdx]->size == 4)
            {
                pcNameChosen = &flcnEvtqInfo.ppEvtqSyms[targetIdx]->name[1];
            }
            else
            {
                if (bOffsetUnique)
                {
                    sprintf(buf, "%s+0x%02x", &flcnEvtqInfo.ppEvtqSyms[targetIdx]->name[1], offset);
                }
                else
                {
                    sprintf(buf, "%s+???", &flcnEvtqInfo.ppEvtqSyms[targetIdx]->name[1]);
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

/*!
 * @brief  Print out the content of msg in the queue
 *
 * @param[in]  pxQueue       The queue structure to be printed out
 * @param[in]  numSpaces     How many spaces are printed out before the content
 */
static void
_flcnXqueuePrintMsg
(
    FLCN_RTOS_XQUEUE *pxQueue,
    LwU32             numSpaces
)
{
    LwU32 wordsRead;

    // Dump the message that is waiting in the queue
    if (pxQueue->messagesWaiting > 0)
    {
        LwU32  idx;
        LwU32  j;
        LwU32  totalSize;
        LwU32  alignedSize;
        LwU8  *pData;
        LwU32  cursor;
        LwU32  bytesDump;
        LwU32  msgBufLen;
        LwU32  dmemSize;
        LwU32  engineBase;

        engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

        dmemSize = thisFlcn->pFCIF->flcnDmemGetSize(engineBase);
        totalSize = pxQueue->length * pxQueue->itemSize;
        alignedSize = (totalSize + 3) >> 2 << 2;
        pData = (LwU8 *)malloc(totalSize + alignedSize);
        cursor = pxQueue->readFrom - pxQueue->head;

        // Validate if the queue structure is valid
        msgBufLen = pxQueue->tail - pxQueue->head;
        if (pxQueue->itemSize * pxQueue->messagesWaiting > msgBufLen)
        {
            dprintf("   ERROR: The queue structure is corrupted, queue.itemSize(%d) * msgWaiting (%d) > queueBufferSize(%d)\n",
                    pxQueue->itemSize, pxQueue->messagesWaiting, msgBufLen);
            return;
        }

        if (totalSize > msgBufLen)
        {
            dprintf("   ERROR: The queue structure is corrupted, queue.itemSize(%d) * queue.length(%d) > queueBufferSize(%d)\n",
                    pxQueue->itemSize, pxQueue->length, msgBufLen);
            return;
        }

        if (pxQueue->head > pxQueue->tail)
        {
            dprintf("   ERROR: The queue structure is corrupted, queue.head(0x%x) > queue.tail(0x%x) \n",
                    pxQueue->head, pxQueue->tail);
        }

        if (msgBufLen > dmemSize)
        {
            dprintf("   ERROR: The queue structure is corrupted, the bufSize(%d) is larger than DMEM size(%d)\n",
                    msgBufLen, dmemSize);
            return;
        }

        if (pxQueue->messagesWaiting > pxQueue->length)
        {
            dprintf("   ERROR: The queue structure is corrupted, the number (%d) is larger than DMEM size(%d)\n",
                    msgBufLen, dmemSize);
            return;
        }

        wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                                  pxQueue->head,
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


/*!
 * @brief  Print the queue in detail view, values of every field will
 *         be printed out
 *
 * @param[in]  pxQueue       The queue structure to be printed out
 * @param[in]  pcQueueName   Name of the queue
 * @param[in]  qAddr         Device address of the queue
 * @param[in]  queueIdx      Index of the queue
 */
static void
_flcnXQueueFullPrint
(
    FLCN_RTOS_XQUEUE *pxQueue,
    const char       *pcQueueName,
    LwU32             qAddr,
    LwU32             queueIdx
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

    _XListPrint(qAddr+offsetof(FLCN_RTOS_XQUEUE, xTasksWaitingToSend),
                "Waiting-To-Send List:",
                TRUE);
    dprintf("lw:\n");
    _XListPrint(qAddr+offsetof(FLCN_RTOS_XQUEUE, xTasksWaitingToReceive),
                "Waiting-To-Receive List:",
                TRUE);
    _flcnXqueuePrintMsg(pxQueue, 7);

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
_flcnXQueuePrint
(
    FLCN_RTOS_XQUEUE *pxQueue,
    const char       *pcQueueName,
    LwU32             qAddr,
    LwU32             queueIdx
)
{
    const char *pcQueueType;
    char        queueNameBuf[128];

    if (pcQueueName == NULL)
    {
        _flcnGetQueueName(qAddr, queueNameBuf, sizeof(queueNameBuf));
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
        dprintf("lw: %3d   %s   %s   %20s\n",
                queueIdx,
                pcQueueType,
                pcSemphrStatus,
                pcQueueName
                );

    }
    else
    {
        pcQueueType = "Q";

        dprintf("lw: %3d   %s   %5d   %20s %6d   %4d   0x%05x   0x%05x   0x%05x   0x%05x\n",
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
        _flcnXqueuePrintMsg(pxQueue, 23);
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
_flcnSymXQueueGetByAddress
(
    LwU32       addr,
    FLCN_RTOS_XQUEUE *pXQueue
)
{
    LwU32 wordsRead;
    LwU32 length = sizeof(FLCN_RTOS_XQUEUE) >> 2;
    LwU32 heapLowest;
    LwU32 heapHighest;
    LwU32 maxMsg;
    LwU32 engineBase;

    engineBase  = thisFlcn->pFEIF->flcnEngGetFalconBase();

    if (!_flcnEvtqLoadSymbol())
    {
        heapLowest = 0;
        heapHighest = thisFlcn->pFCIF->flcnDmemGetSize(engineBase);

        //
        // Set the maximum value to 0xFF which seems to be
        // an impoosible value to reach, but REMEMBER to update
        // if we go more than that
        maxMsg = 0xFF;
    }
    else
    {
        heapLowest = flcnEvtqInfo.heap;
        heapHighest = flcnEvtqInfo.heapEnd;
        maxMsg = heapHighest - heapLowest;
    }

    // read-in the data structure
    wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              addr,             // addr
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

/*!
 * Retrieves the address and size in bytes of the data structure that maps queue
 * IDs to task IDs.
 *
 * @param[out]  pAddr   Out param for address of the data structure
 * @param[out]  pSize   Out param for size of the data structure in bytes
 *
 * @return  LW_TRUE If successful
 */
static LwBool
_flcnGetQueueIdToTaskIdMapAddrAndSize
(
    LwU32 *pAddr,
    LwU32 *pSize
)
{
    LwBool bRet = LW_TRUE;

    LwBool bExactFound;
    LwU32 count;
    FLCN_SYM *pQueueIdToTaskIdMapMatches = flcnSymFind("UcodeQueueIdToTaskIdMap", LW_FALSE, &bExactFound, &count);
    if (!bExactFound)
    {
        // We do not have the global mapping of queue IDs to task IDs, so fail.
        dprintf("ERROR: Running uCode does not have symbol UcodeQueueIdToTaskIdMap. Please try dumping by queue ID or by address directly.\n");
        bRet = LW_FALSE;
        goto _flcnGetQueueIdToTaskIdMapAddrAndSize_exit;
    }

    *pAddr = pQueueIdToTaskIdMapMatches->addr;
    *pSize = pQueueIdToTaskIdMapMatches->size;

_flcnGetQueueIdToTaskIdMapAddrAndSize_exit:
    return bRet;
}

/*!
 * Reads the data structure that maps queue IDs to task IDs from the Falcon's
 * DMEM, given its address and size.
 *
 * @param[out]  ppUcodeQueueIdToTaskIdMap   Out param for the mapping data structure
 * @param[in]   mapAddr                     Address of the data structure
 * @param[in]   mapSizeBytes                Size of the data structure in bytes
 *
 * @return  LW_TRUE If successful
 */
static LwBool
_flcnReadQueueIdToTaskIdMap
(
    LwU8 **ppUcodeQueueIdToTaskIdMap,
    LwU32 mapAddr,
    LwU32 mapSizeBytes
)
{
    const LwU32 engineBase = thisFlcn->pFEIF->flcnEngGetFalconBase();
    const LwU32 mapSizeWords = (mapSizeBytes + sizeof(LwU32) - 1U) / sizeof(LwU32);
    LwU32 wordsRead;

    *ppUcodeQueueIdToTaskIdMap = malloc(mapSizeWords * sizeof(LwU32));
    if (*ppUcodeQueueIdToTaskIdMap == NULL)
    {
        dprintf("ERROR: Could not allocate memory for temporary buffer for UcodeQueueIdToTaskIdMap.\n");
        goto _flcnReadQueueIdToTaskIdMap_fail;
    }

    wordsRead = thisFlcn->pFCIF->flcnDmemRead(engineBase,
                                              mapAddr,
                                              LW_TRUE,
                                              mapSizeWords,
                                              1U,
                                              (LwU32 *)*ppUcodeQueueIdToTaskIdMap);
    if (wordsRead != mapSizeWords)
    {
        dprintf("ERROR: Could not read the UcodeQueueIdToTaskIdMap from the Falcon.\n");
        goto _flcnReadQueueIdToTaskIdMap_fail;
    }

    return LW_TRUE;

_flcnReadQueueIdToTaskIdMap_fail:
    free(*ppUcodeQueueIdToTaskIdMap);
    return LW_FALSE;
}

/*!
 * Gets the data structure that maps queue IDs to task IDs from the Falcon's
 * DMEM.
 *
 * @param[out]  ppUcodeQueueIdToTaskIdMap   Out param for the mapping data structure
 * @param[out]  pNumElements                Out param for number of elements in the data structure
 *
 * @return  LW_TRUE If successful
 */
static LwBool
_flcnGetQueueIdToTaskIdMap
(
    LwU8 **ppUcodeQueueIdToTaskIdMap,
    LwU8 *pNumElements
)
{
    LwBool bRet = LW_TRUE;
    LwU32 mapAddr;
    LwU32 mapSizeBytes;

    bRet = _flcnGetQueueIdToTaskIdMapAddrAndSize(&mapAddr, &mapSizeBytes);
    if (!bRet)
    {
        goto _flcnGetQueueIdToTaskIdMap_exit;
    }

    bRet = _flcnReadQueueIdToTaskIdMap(ppUcodeQueueIdToTaskIdMap, mapAddr, mapSizeBytes);
    if (!bRet)
    {
        goto _flcnGetQueueIdToTaskIdMap_exit;
    }

    *pNumElements =
        (LwU8)(mapSizeBytes + sizeof(**ppUcodeQueueIdToTaskIdMap) - 1U) / sizeof(**ppUcodeQueueIdToTaskIdMap);

_flcnGetQueueIdToTaskIdMap_exit:
    return bRet;
}

/*!
 * Gets the queue ID associated with a task using the task ID
 *
 * @param[in]   qTaskId     Task ID whose queue we want to find the ID of
 * @param[out]  pQueueId    Out param for the task's associated queue ID
 *
 * @return  LW_TRUE If successful
 */
static LwBool
_flcnGetQueueIdFromTaskId
(
    LwU32 qTaskId,
    LwU8 *pQueueId
)
{
    LwU8 *pUcodeQueueIdToTaskIdMap;
    LwU8 numElements;
    LwBool bFound = LW_FALSE;
    LwU8 i;

    if (!_flcnGetQueueIdToTaskIdMap(&pUcodeQueueIdToTaskIdMap, &numElements))
    {
        goto _flcnGetQueueIdFromTaskId_exit;
    }

    //
    // Do a reverse lookup by looking for our task ID, and then return the
    // index/queue ID
    //
    for (i = 0U; i < numElements; i++)
    {
        if (pUcodeQueueIdToTaskIdMap[i] == qTaskId)
        {
            *pQueueId = i;
            bFound = LW_TRUE;
            break;
        }
    }

    if (!bFound)
    {
        dprintf("ERROR: Could not find queue ID for task ID 0x%x in UcodeQueueIdToTaskIdMap.\n", qTaskId);
    }

    free(pUcodeQueueIdToTaskIdMap);

_flcnGetQueueIdFromTaskId_exit:
    return bFound;
}

/*!
 * Gets the address of the address *of* a queue using its queue ID
 *
 * @param[in]   queueId     Queue ID for the queue that we want to retrieve the address of
 * @param[out]  pQueueAddr  Out param for address *of* the queue's address in DMEM
 *
 * @return  LW_TRUE If successful
 */
static LwBool
_flcnGetQueueAddrFromQueueId
(
    LwU8 queueId,
    LwU32 *pQueueAddr
)
{
    const LwLength mapElementSize = sizeof(LwU32);
    LwLength mapNumEntries;

    LwBool bExactFound;
    LwU32 count;
    FLCN_SYM *pQueueIdToTaskIdMapMatches = flcnSymFind("UcodeQueueIdToQueueHandleMap", LW_FALSE, &bExactFound, &count);
    if (!bExactFound)
    {
        // We do not have the global mapping of queue ID to task queue, so fail.
        dprintf("ERROR: Running uCode does not have symbol UcodeQueueIdToQueueHandleMap. \n");
        return LW_FALSE;
    }

    mapNumEntries = pQueueIdToTaskIdMapMatches->size / mapElementSize;
    if (queueId >= mapNumEntries)
    {
        dprintf("ERROR: queue ID 0x%x is not valid in UcodeQueueIdToQueueHandleMap.\n", queueId);
        return LW_FALSE;
    }

    *pQueueAddr = (LwU32)(pQueueIdToTaskIdMapMatches->addr + queueId * mapElementSize);
    return LW_TRUE;
}
