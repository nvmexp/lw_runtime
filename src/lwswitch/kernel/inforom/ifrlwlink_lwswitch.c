/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "error_lwswitch.h"

#include "inforom/inforom_lwswitch.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
LwlStatus
lwswitch_inforom_lwlink_flush
(
    struct lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;
    struct inforom *pInforom = device->pInforom;
    PINFOROM_LWLINK_STATE pLwlinkState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;

    if (pLwlinkState != NULL && pLwlinkState->bDirty)
    {
        status = lwswitch_inforom_write_object(device, "LWL",
                                        pLwlinkState->pFmt, pLwlinkState->pLwl,
                                        pLwlinkState->pPackedObject);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to flush LWL object to InfoROM, rc: %d\n", status);
        }
        else
        {
            pLwlinkState->bDirty = LW_FALSE;
        }
    }

    return status;
}

static void
_inforom_lwlink_get_correctable_error_counts
(
    lwswitch_device                         *device,
    LwU32                                    linkId,
    INFOROM_LWLINK_CORRECTABLE_ERROR_COUNTS *pErrorCounts
)
{
    LwlStatus status;
    LwU32 lane, idx;
    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS p = { 0 };

    ct_assert(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE <=
              INFOROM_LWL_OBJECT_MAX_SUBLINK_WIDTH);

    lwswitch_os_memset(pErrorCounts, 0, sizeof(*pErrorCounts));

    p.linkId = linkId;
    p.counterMask = LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT
                  | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY
                  | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7
                  | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7;

    status = device->hal.lwswitch_ctrl_get_counters(device, &p);
    if (status != LWL_SUCCESS)
    {
        return;
    }

    pErrorCounts->flitCrc =
        p.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)];

    pErrorCounts->txLinkReplay =
        p.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)];

    pErrorCounts->rxLinkReplay =
        p.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)];

    pErrorCounts->linkRecovery =
        p.lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)];

    for (lane = 0; lane < LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE; lane++)
    {
        idx = BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(lane));
        pErrorCounts->laneCrc[lane] = p.lwlinkCounters[idx];
    }
}

static void
_inforom_lwlink_update_correctable_error_rates
(
    lwswitch_device  *device,
    struct inforom   *pInforom

)
{
    PINFOROM_LWLINK_STATE pLwlinkState = pInforom->pLwlinkState;
    LwU64                 enabledLinkMask;
    LwU32                 linkId, publicId, localLinkIdx;
    LwBool                bDirty = LW_FALSE;
    LwBool                bDirtyTemp;
    INFOROM_LWLINK_CORRECTABLE_ERROR_COUNTS errorCounts = { 0 };

    if (pLwlinkState == NULL)
    {
        return;
    }

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, linkId, enabledLinkMask)
    {
        if (device->hal.lwswitch_get_link_public_id(device, linkId, &publicId) != LWL_SUCCESS)
        {
            continue;
        }

        if (device->hal.lwswitch_get_link_local_idx(device, linkId, &localLinkIdx) != LWL_SUCCESS)
        {
            continue;
        }

        _inforom_lwlink_get_correctable_error_counts(device, linkId, &errorCounts);

        if (device->hal.lwswitch_inforom_lwl_update_link_correctable_error_info(device,
                pLwlinkState->pLwl, &pLwlinkState->correctableErrorRateState, linkId,
                publicId, localLinkIdx, &errorCounts, &bDirtyTemp) != LWL_SUCCESS)
        {
            continue;
        }

        bDirty |= bDirtyTemp;
    }
    FOR_EACH_INDEX_IN_MASK_END;

    pLwlinkState->bDirty |= bDirty;
}

static void _lwswitch_lwlink_1hz_callback
(
    lwswitch_device *device
)
{
    struct inforom *pInforom = device->pInforom;

    if ((pInforom == NULL) || (pInforom->pLwlinkState == NULL) ||
        pInforom->pLwlinkState->bCallbackPending)
    {
        return;
    }

    pInforom->pLwlinkState->bCallbackPending = LW_TRUE;
    _inforom_lwlink_update_correctable_error_rates(device, pInforom);
    pInforom->pLwlinkState->bCallbackPending = LW_FALSE;
}

static void
_inforom_lwlink_start_correctable_error_recording
(
    lwswitch_device *device,
    struct inforom  *pInforom
)
{
    PINFOROM_LWLINK_STATE pLwlinkState = pInforom->pLwlinkState;

    if (pLwlinkState == NULL)
    {
        return;
    }

    if (pLwlinkState->bDisableCorrectableErrorLogging)
    {

        LWSWITCH_PRINT(device, INFO,
                "%s: Correctable error recording disabled by regkey or unsupported\n",
                __FUNCTION__);
        return;
    }

    pLwlinkState->bCallbackPending = LW_FALSE;

    lwswitch_task_create(device, &_lwswitch_lwlink_1hz_callback,
                         LWSWITCH_INTERVAL_1SEC_IN_NS, 0);
}

LwlStatus
lwswitch_inforom_lwlink_load
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU8 version = 0;
    LwU8 subversion = 0;
    INFOROM_LWLINK_STATE *pLwlinkState = NULL;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = lwswitch_inforom_get_object_version_info(device, "LWL", &version,
                                                    &subversion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN, "no LWL object found, rc:%d\n", status);
        return LWL_SUCCESS;
    }

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(subversion))
    {
        LWSWITCH_PRINT(device, WARN, "LWL v%u.%u not supported\n",
                    version, subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    LWSWITCH_PRINT(device, INFO, "LWL v%u.%u found\n", version, subversion);

    pLwlinkState = lwswitch_os_malloc(sizeof(INFOROM_LWLINK_STATE));
    if (pLwlinkState == NULL)
    {
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(pLwlinkState, 0, sizeof(INFOROM_LWLINK_STATE));

    pLwlinkState->bDirty = LW_FALSE;
    pLwlinkState->bDisableFatalErrorLogging = LW_FALSE;
    pLwlinkState->bDisableCorrectableErrorLogging = LW_TRUE;

    switch (version)
    {
        case 3:
            pLwlinkState->pFmt = INFOROM_LWL_OBJECT_V3S_FMT;
            pLwlinkState->pPackedObject = lwswitch_os_malloc(INFOROM_LWL_OBJECT_V3S_PACKED_SIZE);
            if (pLwlinkState->pPackedObject == NULL)
            {
                status = -LWL_NO_MEM;
                goto lwswitch_inforom_lwlink_version_fail;
            }

            pLwlinkState->pLwl = lwswitch_os_malloc(sizeof(INFOROM_LWL_OBJECT));
            if (pLwlinkState->pLwl == NULL)
            {
                status = -LWL_NO_MEM;
                lwswitch_os_free(pLwlinkState->pPackedObject);
                goto lwswitch_inforom_lwlink_version_fail;
            }

            pLwlinkState->bDisableCorrectableErrorLogging = LW_FALSE;

            break;

        default:
            LWSWITCH_PRINT(device, WARN, "LWL v%u.%u not supported\n",
                        version, subversion);
            goto lwswitch_inforom_lwlink_version_fail;
            break;
    }

    status = lwswitch_inforom_read_object(device, "LWL", pLwlinkState->pFmt,
                                        pLwlinkState->pPackedObject,
                                        pLwlinkState->pLwl);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to read LWL object, rc:%d\n", status);
        goto lwswitch_inforom_read_fail;
    }

    status = lwswitch_inforom_add_object(pInforom, &pLwlinkState->pLwl->header);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to cache LWL object header, rc:%d\n",
                    status);
        goto lwswitch_inforom_read_fail;
    }

    pInforom->pLwlinkState = pLwlinkState;

    _inforom_lwlink_start_correctable_error_recording(device, pInforom);

    return LWL_SUCCESS;

lwswitch_inforom_read_fail:
    lwswitch_os_free(pLwlinkState->pPackedObject);
    lwswitch_os_free(pLwlinkState->pLwl);
lwswitch_inforom_lwlink_version_fail:
    lwswitch_os_free(pLwlinkState);

    return status;
}

void
lwswitch_inforom_lwlink_unload
(
    lwswitch_device *device
)
{
    INFOROM_LWLINK_STATE *pLwlinkState;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return;
    }

    if (lwswitch_inforom_lwlink_flush(device) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to flush LWL object on object unload\n");
    }

    lwswitch_os_free(pLwlinkState->pPackedObject);
    lwswitch_os_free(pLwlinkState->pLwl);
    lwswitch_os_free(pLwlinkState);
    pInforom->pLwlinkState = NULL;
}

LwlStatus
lwswitch_inforom_lwlink_get_minion_data
(
    lwswitch_device *device,
    LwU8             linkId,
    LwU32           *seedData
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_LWLINK_STATE *pLwlinkState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return device->hal.lwswitch_inforom_lwl_get_minion_data(device,
                                                        pLwlinkState->pLwl,
                                                        linkId, seedData);
}

LwlStatus
lwswitch_inforom_lwlink_set_minion_data
(
    lwswitch_device *device,
    LwU8             linkId,
    LwU32           *seedData,
    LwU32            size
)
{
    LwlStatus status;
    LwBool bDirty;
    struct inforom *pInforom = device->pInforom;
    INFOROM_LWLINK_STATE *pLwlinkState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = device->hal.lwswitch_inforom_lwl_set_minion_data(device,
                                                        pLwlinkState->pLwl,
                                                        linkId, seedData, size,
                                                        &bDirty);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to set minion data, rc:%d\n",
                    status);
    }

    pLwlinkState->bDirty |= bDirty;

    return status;
}

LwlStatus
lwswitch_inforom_lwlink_log_error_event
(
    lwswitch_device            *device,
    void                       *error_event
)
{
    LwlStatus status;
    LwBool bDirty = LW_FALSE;
    struct inforom *pInforom = device->pInforom;
    INFOROM_LWLINK_STATE *pLwlinkState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = device->hal.lwswitch_inforom_lwl_log_error_event(device,
                                                        pLwlinkState->pLwl,
                                                        (INFOROM_LWLINK_ERROR_EVENT *)error_event,
                                                        &bDirty);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to log error to inforom, rc:%d\n",
                    status);
    }

    pLwlinkState->bDirty |= bDirty;

    return status;
}

LwlStatus
lwswitch_inforom_lwlink_get_max_correctable_error_rate
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;

    if ((pInforom == NULL) || (pInforom->pLwlinkState == NULL))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return device->hal.lwswitch_inforom_lwl_get_max_correctable_error_rate(device, params);
}

LwlStatus
lwswitch_inforom_lwlink_get_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;

    if ((pInforom == NULL) || (pInforom->pLwlinkState == NULL))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return device->hal.lwswitch_inforom_lwl_get_errors(device, params);
}
#else
LwlStatus
lwswitch_inforom_lwlink_flush
(
    struct lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwlink_load
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

void
lwswitch_inforom_lwlink_unload
(
    lwswitch_device *device
)
{
    return;
}

LwlStatus
lwswitch_inforom_lwlink_get_minion_data
(
    lwswitch_device *device,
    LwU8             linkId,
    LwU32           *seedData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwlink_set_minion_data
(
    lwswitch_device *device,
    LwU8             linkId,
    LwU32           *seedData,
    LwU32            size
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwlink_log_error_event
(
    lwswitch_device            *device,
    void                       *error_event
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwlink_get_max_correctable_error_rate
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwlink_get_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
