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
#include "lr10/lr10.h"
#include "flcn/flcn_lwswitch.h"

#include "lwswitch/lr10/dev_falcon_v4.h"

static LwU32
_flcnRegRead_LR10
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
_flcnRegWrite_LR10
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

/*
 * @brief Retrigger an interrupt message from the engine to the LW_CTRL tree
 *
 * @param[in] device  lwswitch_device pointer
 * @param[in] pFlcn   FLCN pointer
 */
static void
_flcnIntrRetrigger_LR10
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    LwU32 val = DRF_DEF(_PFALCON, _FALCON_INTR_RETRIGGER, _TRIGGER, _TRUE);
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_INTR_RETRIGGER(0), val);
}

static LwBool
_flcnAreEngDescsInitialized_LR10
(
    lwswitch_device    *device,
    FLCN               *pFlcn
)
{
    // if pFlcn->engDeslwc is 0, we haven't finished discovery, return false
    // if pFlcn->engDeslwc is NOT 0, and pFlcn->engDescBc is NULL, this is a unicast only engine
    return   pFlcn->engDeslwc.base != 0 && pFlcn->engDeslwc.initialized &&
            (pFlcn->engDescBc.base == 0 || pFlcn->engDescBc.initialized);
}

/*
 *  @brief Waits for falcon to finish scrubbing IMEM/DMEM.
 *
 *  @param[in] device   switch device
 *  @param[in] pFlcn    FLCN pointer
 *
 *  @returns nothing
 */
static LW_STATUS
_flcnWaitForResetToFinish_LR10
(
    lwswitch_device    *device,
    PFLCN               pFlcn
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU32 dmaCtrl;

    // Add a dummy write (of anything) to trigger scrubbing
    flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_MAILBOX0, 0);

    // TODO: Adapt timeout to our model, this should be centralized.
    if (IS_EMULATION(device))
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }

    while (1)
    {
        dmaCtrl = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_DMACTL);

        if (FLD_TEST_DRF(_PFALCON, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, dmaCtrl) &&
            FLD_TEST_DRF(_PFALCON, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, dmaCtrl))
        {
            // Operation successful, IMEM and DMEM scrubbing has finished.
            return LW_OK;                    
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for scrubbing to finish!!!\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            return LW_ERR_TIMEOUT;
        }
    }
}

/*!
 * @brief   Capture and dump the falconPC trace.
 *
 * @param[in]  device     lwswitch device pointer
 * @param[in]  pFlcn      FLCN object pointer
 *
 * @returns nothing
 */
void
_flcnDbgInfoCapturePcTrace_LR10
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32    regTraceIdx;
    LwU32    idx;
    LwU32    maxIdx;

    // Dump entire PC trace buffer
    regTraceIdx = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_TRACEIDX);
    maxIdx      = DRF_VAL(_PFALCON_FALCON, _TRACEIDX, _MAXIDX, regTraceIdx);

    LWSWITCH_PRINT(device, ERROR,
              "PC TRACE (TOTAL %d ENTRIES. Entry 0 is the most recent branch):\n",
              maxIdx);

    for (idx = 0; idx < maxIdx; idx++)
    {
        regTraceIdx =
            FLD_SET_DRF_NUM(_PFALCON, _FALCON_TRACEIDX, _IDX, idx, regTraceIdx);

        flcnRegWrite_HAL(device, pFlcn, LW_PFALCON_FALCON_TRACEIDX, regTraceIdx);

        LWSWITCH_PRINT(device, ERROR, "FALCON_TRACEPC(%d)     : 0x%08x\n", idx,
            DRF_VAL(_PFALCON, _FALCON_TRACEPC, _PC,
                flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_TRACEPC)));
    }
}

/*!
 * @brief Read falcon core revision
 *
 * @param[in] device lwswitch_device pointer
 * @param[in] pFlcn  FLCN pointer
 *
 * @return @ref LW_FLCN_CORE_REV_X_Y.
 */
LwU8
_flcnReadCoreRev_LR10
(
    lwswitch_device *device,
    PFLCN            pFlcn
)
{
    LwU32 hwcfg1 = flcnRegRead_HAL(device, pFlcn, LW_PFALCON_FALCON_HWCFG1);

    return ((DRF_VAL(_PFALCON, _FALCON_HWCFG1, _CORE_REV, hwcfg1) << 4) |
            DRF_VAL(_PFALCON, _FALCON_HWCFG1, _CORE_REV_SUBVERSION, hwcfg1));
}

//
// Store pointers to ucode header and data.
// Preload ucode from registry if available.
//
LW_STATUS
_flcnConstruct_LR10
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
            goto _flcnConstruct_LR10_fail;
        }
        lwswitch_os_memset(pQueueInfo, 0, sizeof(FALCON_QUEUE_INFO));
        // Assert if Number of Queues are zero
        LWSWITCH_ASSERT(pFlcn->numQueues != 0);
        pQueueInfo->pQueues = lwswitch_os_malloc(sizeof(FLCNQUEUE) * pFlcn->numQueues);
        if (pQueueInfo->pQueues == NULL)
        {
            status = LW_ERR_NO_MEMORY;
            LWSWITCH_ASSERT(0);
            goto _flcnConstruct_LR10_fail;
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
                goto _flcnConstruct_LR10_fail;
            }
            flcnQueueSeqInfoStateInit(device, pFlcn);
        }
    }
    // DEBUG
    LWSWITCH_PRINT(device, INFO, "Falcon: %s\n", flcnGetName_HAL(device, pFlcn));
    LWSWITCH_ASSERT(pFlcnable != NULL);
    flcnableGetExternalConfig(device, pFlcnable, &pFlcn->extConfig);
    return LW_OK;
_flcnConstruct_LR10_fail:
    // call flcnDestruct to free the memory allocated in this construct function
    flcnDestruct_HAL(device, pFlcn);
    return status;
}

void
_flcnDestruct_LR10
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
_flcnGetName_LR10
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

/**
 * @brief   set hal function pointers for functions defined in LR10 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_LR10
(
    PFLCN            pFlcn
)
{
    flcn_hal *pHal = pFlcn->pHal;

    pHal->readCoreRev              = _flcnReadCoreRev_LR10;
    pHal->regRead                  = _flcnRegRead_LR10;
    pHal->regWrite                 = _flcnRegWrite_LR10;
    pHal->construct                = _flcnConstruct_LR10;
    pHal->destruct                 = _flcnDestruct_LR10;
    pHal->getName                  = _flcnGetName_LR10;
    pHal->intrRetrigger            = _flcnIntrRetrigger_LR10;
    pHal->areEngDescsInitialized   = _flcnAreEngDescsInitialized_LR10;
    pHal->waitForResetToFinish     = _flcnWaitForResetToFinish_LR10;
    pHal->dbgInfoCapturePcTrace    = _flcnDbgInfoCapturePcTrace_LR10;
}
