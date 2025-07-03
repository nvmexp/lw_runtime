/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "rmflcncmdif_lwswitch.h"

#include "common_lwswitch.h"
#include "lwstatus.h"

/*!
 * @brief Read the falcon core revision and subversion.
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcnable   FLCNABLE object pointer
 *
 * @return @ref LW_FLCN_CORE_REV_X_Y.
 */
static LwU8
_flcnableReadCoreRev_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable
)
{
    return flcnReadCoreRev_HAL(device, pFlcnable->pFlcn);
}

/*!
 * @brief Get external config
 */
static void
_flcnableGetExternalConfig_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    PFALCON_EXTERNAL_CONFIG pConfig
)
{
    pConfig->bResetInPmc = LW_FALSE;
    pConfig->blkcgBase = 0xffffffff;
    pConfig->fbifBase = 0xffffffff;
}

/*!
 * @brief   Retrieve content from falcon's EMEM.
 */
static void
_flcnableEmemCopyFrom_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            src,
    LwU8            *pDst,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_PRINT(device, ERROR,
        "%s: FLCNABLE interface not implemented on this falcon!\n",
        __FUNCTION__);
    LWSWITCH_ASSERT(0);
}

/*!
 * @brief   Write content to falcon's EMEM.
 */
static void
_flcnableEmemCopyTo_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            dst,
    LwU8            *pSrc,
    LwU32            sizeBytes,
    LwU8             port
)
{
    LWSWITCH_PRINT(device, ERROR,
        "%s: FLCNABLE interface not implemented on this falcon!\n",
        __FUNCTION__);
    LWSWITCH_ASSERT(0);
}

/*
 * @brief Handle INIT Event
 */
static LW_STATUS
_flcnableHandleInitEvent_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    RM_FLCN_MSG     *pGenMsg
)
{
    return LW_OK;
}

/*!
 * @brief   Retrieves a pointer to the engine specific SEQ_INFO structure.
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcnable   FLCNABLE object pointer
 * @param[in]   seqIndex    Index of the structure to retrieve
 *
 * @return  Pointer to the SEQ_INFO structure or NULL on invalid index.
 */
static PFLCN_QMGR_SEQ_INFO
_flcnableQueueSeqInfoGet_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    LwU32            seqIndex
)
{
    LWSWITCH_PRINT(device, ERROR,
        "%s: FLCNABLE interface not implemented on this falcon!\n",
        __FUNCTION__);
    LWSWITCH_ASSERT(0);
    return NULL;
}

/*!
 * @brief   Clear out the engine specific portion of the SEQ_INFO structure.
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcnable   FLCNABLE object pointer
 * @param[in]   pSeqInfo    SEQ_INFO structure pointer
 */
static void
_flcnableQueueSeqInfoClear_IMPL
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
}

/*!
 * @brief   Free up all the engine specific sequence allocations.
 *
 * @param[in]   device      lwswitch device pointer
 * @param[in]   pFlcnable   FLCNABLE object pointer
 * @param[in]   pSeqInfo    SEQ_INFO structure pointer
 */
static void
_flcnableQueueSeqInfoFree_IMPL
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
}

/*!
 * @brief   Validate that the given CMD and related params are properly formed.
 *
 * @copydoc flcnQueueCmdPostNonBlocking_IMPL
 *
 * @return  Boolean if command was properly formed.
 */
static LwBool
_flcnableQueueCmdValidate_IMPL
(
    lwswitch_device *device,
    PFLCNABLE        pFlcnable,
    PRM_FLCN_CMD     pCmd,
    PRM_FLCN_MSG     pMsg,
    void            *pPayload,
    LwU32            queueIdLogical
)
{
    LWSWITCH_PRINT(device, ERROR,
        "%s: FLCNABLE interface not implemented on this falcon!\n",
        __FUNCTION__);
    LWSWITCH_ASSERT(0);
    return LW_FALSE;
}

/*!
 * @brief   Engine specific command post actions.
 *
 * @copydoc flcnQueueCmdPostNonBlocking_IMPL
 *
 * @return  LW_OK on success
 *          Failure specific error codes
 */
static LW_STATUS
_flcnableQueueCmdPostExtension_IMPL
(
    lwswitch_device    *device,
    PFLCNABLE           pFlcnable,
    PRM_FLCN_CMD        pCmd,
    PRM_FLCN_MSG        pMsg,
    void               *pPayload,
    LWSWITCH_TIMEOUT   *pTimeout,
    PFLCN_QMGR_SEQ_INFO pSeqInfo
)
{
    return LW_OK;
}

static void
_flcnablePostDiscoveryInit_IMPL
(
    lwswitch_device *device,
    FLCNABLE        *pSoe
)
{
    flcnPostDiscoveryInit(device, pSoe->pFlcn);
}

/**
 * @brief   sets pEngDeslwc and pEngDescBc to the discovered
 * engine that matches this flcnable instance
 *
 * @param[in]   device       lwswitch_device pointer
 * @param[in]   pSoe         SOE pointer
 * @param[out]  pEngDeslwc  pointer to the UniCast Engine
 *       Descriptor Pointer
 * @param[out]  pEngDescBc  pointer to the BroadCast Engine
 *       Descriptor Pointer
 */
static void
_flcnableFetchEngines_IMPL
(
    lwswitch_device         *device,
    FLCNABLE                *pSoe,
    ENGINE_DESCRIPTOR_TYPE  *pEngDeslwc,
    ENGINE_DESCRIPTOR_TYPE  *pEngDescBc
)
{
    // Every falcon REALLY needs to implement this. If they don't flcnRegRead and flcnRegWrite won't work
    LWSWITCH_PRINT(device, ERROR,
        "%s: FLCNABLE interface not implemented on this falcon!\n",
        __FUNCTION__);
    LWSWITCH_ASSERT(0);
}


/* -------------------- Object construction/initialization ------------------- */
static void
flcnableSetupHal
(
    FLCNABLE *pFlcnable,
    LwU32     pci_device_id
)
{
    flcnable_hal *pHal = pFlcnable->pHal;

    //init hal Interfaces
    pHal->readCoreRev                             = _flcnableReadCoreRev_IMPL;
    pHal->getExternalConfig                       = _flcnableGetExternalConfig_IMPL;
    pHal->ememCopyFrom                            = _flcnableEmemCopyFrom_IMPL;
    pHal->ememCopyTo                              = _flcnableEmemCopyTo_IMPL;
    pHal->handleInitEvent                         = _flcnableHandleInitEvent_IMPL;
    pHal->queueSeqInfoGet                         = _flcnableQueueSeqInfoGet_IMPL;
    pHal->queueSeqInfoClear                       = _flcnableQueueSeqInfoClear_IMPL;
    pHal->queueSeqInfoFree                        = _flcnableQueueSeqInfoFree_IMPL;
    pHal->queueCmdValidate                        = _flcnableQueueCmdValidate_IMPL;
    pHal->queueCmdPostExtension                   = _flcnableQueueCmdPostExtension_IMPL;
    pHal->postDiscoveryInit                       = _flcnablePostDiscoveryInit_IMPL;
    pHal->fetchEngines                            = _flcnableFetchEngines_IMPL;
}

LwlStatus
flcnableInit
(
    lwswitch_device    *device,
    FLCNABLE           *pFlcnable,
    LwU32               pci_device_id
)
{
    LwlStatus retval;
    FLCN *pFlcn = NULL;

    // allocate hal if a child class hasn't already
    if (pFlcnable->pHal == NULL)
    {
        flcnable_hal *pHal = pFlcnable->pHal = lwswitch_os_malloc(sizeof(*pHal));
        if (pHal == NULL)
        {
            LWSWITCH_PRINT(device, ERROR, "Flcn allocation failed!\n");
            retval = -LWL_NO_MEM;
            goto flcnable_init_fail;
        }
        lwswitch_os_memset(pHal, 0, sizeof(*pHal));
    }

    // init flcn - a little out of place here, since we're really only
    // supposed to be initializing hals. However, we need pci_device_id
    // to initialize flcn's hals and flcn is _very_ closely tied to
    // flcnable so it kind of makes some sense to allocate it here
    pFlcn = pFlcnable->pFlcn = flcnAllocNew();
    if (pFlcn == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "Flcn allocation failed!\n");
        retval = -LWL_NO_MEM;
        goto flcnable_init_fail;
    }
    retval = flcnInit(device, pFlcn, pci_device_id);
    if (retval != LWL_SUCCESS)
    {
        goto flcnable_init_fail;
    }

    //don't have a parent class to init, go straight to setupHal
    flcnableSetupHal(pFlcnable, pci_device_id);

    return retval;

flcnable_init_fail:
    flcnableDestroy(device, pFlcnable);
    return retval;
}

// reverse of flcnableInit()
void
flcnableDestroy
(
    lwswitch_device    *device,
    FLCNABLE           *pFlcnable
)
{
    if (pFlcnable->pFlcn != NULL)
    {
        flcnDestroy(device, pFlcnable->pFlcn);
        lwswitch_os_free(pFlcnable->pFlcn);
        pFlcnable->pFlcn = NULL;
    }

    if (pFlcnable->pHal != NULL)
    {
        lwswitch_os_free(pFlcnable->pHal);
        pFlcnable->pHal = NULL;
    }
}
