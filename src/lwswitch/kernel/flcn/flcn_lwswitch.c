/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"

#include "flcn/flcn_lwswitch.h"
#include "flcn/flcnable_lwswitch.h"

static void flcnSetupIpHal(lwswitch_device *device, PFLCN pFlcn);

/*!
 * @brief   Get the falcon core revision and subversion.
 *
 * @param[in]   device  lwswitch device pointer
 * @param[in]   pFlcn   FLCN object pointer
 *
 * @return the falcon core revision in the format of LW_FLCN_CORE_REV_X_Y.
 */
static LwU8
_flcnCoreRevisionGet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    if (pFlcn->coreRev == 0x00)
    {
        // Falcon core revision has not yet been set.  Set it now.
        flcnGetCoreInfo_HAL(device, pFlcn);
    }

    return pFlcn->coreRev;
}

/*!
 *  @brief Mark the falcon as not ready and inaccessible from RM.
 *  osHandleGpuSurpriseRemoval will use this routine to prevent access to the
 *  Falcon, which could crash due to absense of GPU, during driver cleanup.
 *
 *  @param[in] device lwswitch_device pointer
 *  @param[in] pFlcn  FLCN pointer
 *
 *  @returns nothing
 */
static void
_flcnMarkNotReady_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    pFlcn->bOSReady = LW_FALSE;
}

/*!
 * Retrieves the current head pointer for given physical command queue index.
 *
 * @param[in]   device  lwswitch device pointer
 * @param[in]   pFlcn   FLCN object pointer
 * @param[in]   pQueue  Pointer to the queue
 * @param[out]  pHead   Pointer to write with the queue's head pointer
 *
 * @return 'LW_OK' if head value was successfully retrieved.
 */
static LW_STATUS
_flcnCmdQueueHeadGet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pHead
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->cmdQHeadSize);
    LWSWITCH_ASSERT(pHead != NULL);

    *pHead = flcnRegRead_HAL(device, pFlcn,
                                    (pQueueInfo->cmdQHeadBaseAddress +
                                    (pQueue->queuePhyId * pQueueInfo->cmdQHeadStride)));
    return LW_OK;
}

/*!
 * Sets the head pointer for the given physical command queue index.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  head    The desired head value for the queue
 *
 * @return 'LW_OK' if the head value was successfully set.
 */
static LW_STATUS
_flcnCmdQueueHeadSet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            head
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->cmdQHeadSize);

    flcnRegWrite_HAL(device, pFlcn,
                            (pQueueInfo->cmdQHeadBaseAddress +
                            (pQueue->queuePhyId * pQueueInfo->cmdQHeadStride)),
                            head);
    return LW_OK;
}

/*!
 * Retrieves the current tail pointer for given physical command queue index.
 *
 * @param[in]   device  lwswitch device pointer
 * @param[in]   pFlcn   FLCN object pointer
 * @param[in]   pQueue  Pointer to the queue
 * @param[out]  pTail   Pointer to write with the queue's tail value
 *
 * @return 'LW_OK' if the tail value was successfully retrieved.
 */
static LW_STATUS
_flcnCmdQueueTailGet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pTail
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->cmdQTailSize);
    LWSWITCH_ASSERT(pTail != NULL);

    *pTail = flcnRegRead_HAL(device, pFlcn,
                                    (pQueueInfo->cmdQTailBaseAddress +
                                    (pQueue->queuePhyId * pQueueInfo->cmdQTailStride)));
    return LW_OK;
}

/*!
 * Set the Command Queue tail pointer.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  tail    The desired tail value
 *
 * @return 'LW_OK' if the tail value was successfully set.
 */
static LW_STATUS
_flcnCmdQueueTailSet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            tail
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->cmdQTailSize);

    flcnRegWrite_HAL(device, pFlcn,
                            (pQueueInfo->cmdQTailBaseAddress +
                            (pQueue->queuePhyId * pQueueInfo->cmdQTailStride)),
                            tail);
    return LW_OK;
}

/*!
 * Retrieve the current Message Queue Head pointer.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  pHead   Pointer to write with the queue's head value
 *
 * @return 'LW_OK' if the queue's head value was successfully retrieved.
 */
static LW_STATUS
_flcnMsgQueueHeadGet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pHead
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->msgQHeadSize);
    LWSWITCH_ASSERT(pHead != NULL);

    *pHead = flcnRegRead_HAL(device, pFlcn,
                                    (pQueueInfo->msgQHeadBaseAddress +
                                    (pQueue->queuePhyId * pQueueInfo->msgQHeadStride)));
    return LW_OK;
}

/*!
 * Set the Message Queue Head pointer.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  head    The desired head value
 *
 * @return 'LW_OK' if the head value was successfully set.
 */
static LW_STATUS
_flcnMsgQueueHeadSet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            head
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->msgQHeadSize);

    flcnRegWrite_HAL(device, pFlcn,
                            (pQueueInfo->msgQHeadBaseAddress +
                            (pQueue->queuePhyId * pQueueInfo->msgQHeadStride)),
                            head);
    return LW_OK;
}

/*!
 * Retrieve the current Message Queue Tail pointer.
 *
 * @param[in]   device  lwswitch device pointer
 * @param[in]   pFlcn   FLCN object pointer
 * @param[in]   pQueue  Pointer to the queue
 * @param[out]  pTail   Pointer to write with the message queue's tail value
 *
 * @return 'LW_OK' if the tail value was successfully retrieved.
 */
static LW_STATUS
_flcnMsgQueueTailGet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32           *pTail
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->msgQTailSize);
    LWSWITCH_ASSERT(pTail != NULL);

    *pTail = flcnRegRead_HAL(device, pFlcn,
                                    (pQueueInfo->msgQTailBaseAddress +
                                    (pQueue->queuePhyId * pQueueInfo->msgQTailStride)));
    return LW_OK;
}

/*!
 * Set the Message Queue Tail pointer.
 *
 * @param[in]  device  lwswitch device pointer
 * @param[in]  pFlcn   FLCN object pointer
 * @param[in]  pQueue  Pointer to the queue
 * @param[in]  tail    The desired tail value for the message queue
 *
 * @return 'LW_OK' if the tail value was successfully set.
 */
static LW_STATUS
_flcnMsgQueueTailSet_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    FLCNQUEUE       *pQueue,
    LwU32            tail
)
{
    PFALCON_QUEUE_INFO  pQueueInfo = pFlcn->pQueueInfo;

    LWSWITCH_ASSERT(pQueueInfo != NULL);
    LWSWITCH_ASSERT(pQueue->queuePhyId < pQueueInfo->msgQTailSize);

    flcnRegWrite_HAL(device, pFlcn,
                            (pQueueInfo->msgQTailBaseAddress +
                            (pQueue->queuePhyId * pQueueInfo->msgQTailStride)),
                            tail);
    return LW_OK;
}

/*!
 * Copies 'sizeBytes' from DMEM offset 'src' to 'pDst' using DMEM access
 * port 'port'.
 *
 * @param[in]   device     lwswitch device pointer
 * @param[in]   pFlcn      FLCN pointer
 * @param[in]   src        The DMEM offset for the source of the copy
 * @param[out]  pDst       Pointer to write with copied data from DMEM
 * @param[in]   sizeBytes  The number of bytes to copy from DMEM
 * @param[in]   port       The DMEM port index to use when accessing the DMEM
 */
static LW_STATUS
_flcnDmemCopyFrom_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            src,
    LwU8            *pDst,
    LwU32            sizeBytes,
    LwU8             port
)
{
    return flcnDmemTransfer_HAL(device, pFlcn,
                                src, pDst, sizeBytes, port,
                                LW_TRUE);                   // bCopyFrom
}

/*!
 * Copies 'sizeBytes' from 'pDst' to DMEM offset 'dst' using DMEM access port
 * 'port'.
 *
 * @param[in]  device     lwswitch device pointer
 * @param[in]  pFlcn      FLCN pointer
 * @param[in]  dst        The destination DMEM offset for the copy
 * @param[in]  pSrc       The pointer to the buffer containing the data to copy
 * @param[in]  sizeBytes  The number of bytes to copy into DMEM
 * @param[in]  port       The DMEM port index to use when accessing the DMEM
 */
static LW_STATUS
_flcnDmemCopyTo_IMPL
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwU8             port
)
{
    return flcnDmemTransfer_HAL(device, pFlcn,
                                dst, pSrc, sizeBytes, port,
                                LW_FALSE);                  // bCopyFrom
}

static void
_flcnPostDiscoveryInit_IMPL
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    flcnableFetchEngines_HAL(device, pFlcn->pFlcnable, &pFlcn->engDeslwc, &pFlcn->engDescBc);

    flcnSetupIpHal(device, pFlcn);
}

/* -------------------- Object construction/initialization ------------------- */

/**
 * @brief   set hal object-interface function pointers to flcn implementations
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
static void
flcnSetupHal
(
    PFLCN pFlcn,
    LwU32 pci_device_id
)
{
    flcn_hal *pHal = NULL;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
    if (lwswitch_is_sv10_device_id(pci_device_id))
    {
        flcnSetupHal_SV10(pFlcn);
        goto _flcnSetupHal_success;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    if (lwswitch_is_lr10_device_id(pci_device_id))
    {
        flcnSetupHal_LR10(pFlcn);
        goto _flcnSetupHal_success;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (lwswitch_is_ls10_device_id(pci_device_id))
    {
        flcnSetupHal_LS10(pFlcn);
        goto _flcnSetupHal_success;
    }
#endif

    LWSWITCH_PRINT(NULL, ERROR,
        "Flcn hal can't be setup due to unknown device id\n");
    LWSWITCH_ASSERT(0);

_flcnSetupHal_success:
    //init hal OBJ Interfaces
    pHal = pFlcn->pHal;

    pHal->coreRevisionGet         = _flcnCoreRevisionGet_IMPL;
    pHal->markNotReady            = _flcnMarkNotReady_IMPL;
    pHal->cmdQueueHeadGet         = _flcnCmdQueueHeadGet_IMPL;
    pHal->msgQueueHeadGet         = _flcnMsgQueueHeadGet_IMPL;
    pHal->cmdQueueTailGet         = _flcnCmdQueueTailGet_IMPL;
    pHal->msgQueueTailGet         = _flcnMsgQueueTailGet_IMPL;
    pHal->cmdQueueHeadSet         = _flcnCmdQueueHeadSet_IMPL;
    pHal->msgQueueHeadSet         = _flcnMsgQueueHeadSet_IMPL;
    pHal->cmdQueueTailSet         = _flcnCmdQueueTailSet_IMPL;
    pHal->msgQueueTailSet         = _flcnMsgQueueTailSet_IMPL;

    pHal->dmemCopyFrom            = _flcnDmemCopyFrom_IMPL;
    pHal->dmemCopyTo              = _flcnDmemCopyTo_IMPL;
    pHal->postDiscoveryInit       = _flcnPostDiscoveryInit_IMPL;

    flcnQueueSetupHal(pFlcn);
    flcnRtosSetupHal(pFlcn);
    flcnQueueRdSetupHal(pFlcn);
}

static void
flcnSetupIpHal
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU8 coreRev = flcnableReadCoreRev(device, pFlcn->pFlcnable);

    switch (coreRev) {
        case LW_FLCN_CORE_REV_3_0:
        {
            flcnSetupHal_v03_00(pFlcn);
            break;
        }
        case LW_FLCN_CORE_REV_4_0:
        case LW_FLCN_CORE_REV_4_1:
        {
            flcnSetupHal_v04_00(pFlcn);
            break;
        }
        case LW_FLCN_CORE_REV_5_0:
        case LW_FLCN_CORE_REV_5_1:
        {
            flcnSetupHal_v05_01(pFlcn);
            break;
        }
        case LW_FLCN_CORE_REV_6_0:
        {
            flcnSetupHal_v06_00(pFlcn);
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unsupported falcon core revision: %hhu!\n",
                __FUNCTION__, coreRev);
            LWSWITCH_ASSERT(0);
            break;
        }
    }
}

FLCN *
flcnAllocNew(void)
{
    FLCN *pFlcn = lwswitch_os_malloc(sizeof(*pFlcn));
    if (pFlcn != NULL)
    {
        lwswitch_os_memset(pFlcn, 0, sizeof(*pFlcn));
    }

    return pFlcn;
}

LwlStatus
flcnInit
(
    lwswitch_device    *device,
    FLCN               *pFlcn,
    LwU32               pci_device_id
)
{
    LwlStatus retval = LWL_SUCCESS;

    // allocate hal if a child class hasn't already
    if (pFlcn->pHal == NULL)
    {
        flcn_hal *pHal = pFlcn->pHal = lwswitch_os_malloc(sizeof(*pHal));
        if (pHal == NULL)
        {
            LWSWITCH_PRINT(device, ERROR, "Flcn allocation failed!\n");
            retval = -LWL_NO_MEM;
            goto flcn_init_fail;
        }
        lwswitch_os_memset(pHal, 0, sizeof(*pHal));
    }

    //don't have a parent class to init, go straight to setupHal
    flcnSetupHal(pFlcn, pci_device_id);

    return retval;

flcn_init_fail:
    flcnDestroy(device, pFlcn);
    return retval;
}

// reverse of flcnInit()
void
flcnDestroy
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    if (pFlcn->pHal != NULL)
    {
        lwswitch_os_free(pFlcn->pHal);
        pFlcn->pHal = NULL;
    }
}
