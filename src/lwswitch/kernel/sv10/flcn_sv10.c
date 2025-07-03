/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwmisc.h"

#include "lwswitch/lr10/dev_falcon_v4.h"

#include "common_lwswitch.h"
#include "flcn/flcn_lwswitch.h"

/*!
 * @brief Read falcon core revision
 *
 * @param[in] device lwswitch_device pointer
 * @param[in] pFlcn  FLCN pointer
 *
 * @return @ref LW_FLCN_CORE_REV_X_Y.
 */
LwU8
flcnReadCoreRev_SV10
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32 hwcfg1 = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_HWCFG1);

    return ((DRF_VAL(_PFALCON, _FALCON_HWCFG1, _CORE_REV, hwcfg1) << 4) |
            DRF_VAL(_PFALCON, _FALCON_HWCFG1, _CORE_REV_SUBVERSION, hwcfg1));
}

static LwU32
_flcnRegRead_SV10
(
    lwswitch_device *device,
    PFLCN            pFlcn,
    LwU32            offset
)
{
    // Probably should perform some checks on the offset, the device, and the engine descriptor
    return lwswitch_reg_read_32(device, pFlcn->engDeslwc.base + offset);
}

static void
_flcnRegWrite_SV10
(
    lwswitch_device    *device,
    PFLCN               pFlcn,
    LwU32               offset,
    LwU32               data
)
{
    // Probably should perform some checks on the offset, the device, and the engine descriptor
    lwswitch_reg_write_32(device, pFlcn->engDeslwc.base + offset, data);
}

//
// Store pointers to ucode header and data.
// Preload ucode from registry if available.
//
LW_STATUS
flcnConstruct_SV10
(
    lwswitch_device    *device,
    PFLCN               pFlcn
)
{
    LW_STATUS          status;
    PFLCNABLE          pFlcnable = pFlcn->pFlcnable;
    PFALCON_QUEUE_INFO pQueueInfo;

    pFlcn->bConstructed         = LW_TRUE;

    if (pFlcn->engArch == LW_UPROC_ENGINE_ARCH_DEFAULT)
    {
        // Default the arch to Falcon if it's not set
        pFlcn->engArch = LW_UPROC_ENGINE_ARCH_FALCON;
    }

    // Allocate the memory for Queue Data Structure if needed.
    if (pFlcn->bQueuesEnabled)
    {
        pQueueInfo = pFlcn->pQueueInfo = lwswitch_os_malloc(sizeof(*pQueueInfo));
        if (pQueueInfo == NULL)
        {
            status = LW_ERR_NO_MEMORY;
            LWSWITCH_ASSERT(0);
            goto flcnConstruct_SV10_fail;
        }

        lwswitch_os_memset(pQueueInfo, 0, sizeof(FALCON_QUEUE_INFO));

        // Assert if Number of Queues are zero
        LWSWITCH_ASSERT(pFlcn->numQueues != 0);

        pQueueInfo->pQueues = lwswitch_os_malloc(sizeof(FLCNQUEUE) * pFlcn->numQueues);

        if (pQueueInfo->pQueues == NULL)
        {
            status = LW_ERR_NO_MEMORY;
            LWSWITCH_ASSERT(0);
            goto flcnConstruct_SV10_fail;
        }

        lwswitch_os_memset(pQueueInfo->pQueues, 0, sizeof(FLCNQUEUE) * pFlcn->numQueues);

        // Sequences can be optional
        if (pFlcn->numSequences != 0)
        {
            if ((pFlcn->numSequences - 1) > ((LwU32)LW_U8_MAX))
            {
                status = LW_ERR_OUT_OF_RANGE;
                LWSWITCH_PRINT(device, ERROR,
                          "Max numSequences index = %d cannot fit into byte\n",
                          (pFlcn->numSequences - 1));
                LWSWITCH_ASSERT(0);
                goto flcnConstruct_SV10_fail;
            }

            flcnQueueSeqInfoStateInit(device, pFlcn);
        }
    }

    // DEBUG
    LWSWITCH_PRINT(device, INFO, "Falcon: %s\n", flcnGetName_HAL(device, pFlcn));

    LWSWITCH_ASSERT(pFlcnable != NULL);

    flcnableGetExternalConfig(device, pFlcnable, &pFlcn->extConfig);

    return LW_OK;

flcnConstruct_SV10_fail:

    // call flcnDestruct to free the memory allocated in this construct function
    flcnDestruct_HAL(device, pFlcn);
    return status;
}

void
flcnDestruct_SV10
(
    lwswitch_device    *device,
    PFLCN               pFlcn
)
{
    PFALCON_QUEUE_INFO pQueueInfo;
    PFLCNABLE pFlcnable = pFlcn->pFlcnable;

    if (!pFlcn->bConstructed)
    {
        return;
    }

    pFlcn->bConstructed = LW_FALSE;

    if (pFlcnable == NULL) {
        LWSWITCH_ASSERT(pFlcnable != NULL);
        return;
    }

    if (pFlcn->bQueuesEnabled && (pFlcn->pQueueInfo != NULL))
    {
        pQueueInfo = pFlcn->pQueueInfo;

        if (NULL != pQueueInfo->pQueues)
        {
            lwswitch_os_free(pQueueInfo->pQueues);
            pQueueInfo->pQueues = NULL;
        }

        lwswitch_os_free(pFlcn->pQueueInfo);
        pFlcn->pQueueInfo = NULL;
    }
}

const char *
flcnGetName_SV10
(
    lwswitch_device    *device,
    PFLCN               pFlcn
)
{
    if (pFlcn->name == NULL)
    {
        return "UNKNOWN";
    }
    return pFlcn->name;
}

static LwBool
_flcnAreEngDescsInitialized_SV10
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    // if pFlcn->engDeslwc is zero, we haven't finished discovery, return false
    // ignore pFlcn->engDescBc - SV10 doesn't have separate UC and BC engine descriptors
    return   pFlcn->engDeslwc.base != 0 && pFlcn->engDeslwc.initialized;
}


/**
 * @brief   set hal function pointers for functions defined in SV10 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_SV10
(
    PFLCN            pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    pHal->readCoreRev              =  flcnReadCoreRev_SV10;
    pHal->regRead                  = _flcnRegRead_SV10;
    pHal->regWrite                 = _flcnRegWrite_SV10;
    pHal->construct                =  flcnConstruct_SV10;
    pHal->destruct                 =  flcnDestruct_SV10;
    pHal->getName                  =  flcnGetName_SV10;
    pHal->areEngDescsInitialized   = _flcnAreEngDescsInitialized_SV10;
}
