/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "error_lwswitch.h"

#include "inforom/inforom_lwswitch.h"

LwlStatus
lwswitch_inforom_oms_get_device_disable
(
    lwswitch_device *device,
    LwBool *pBDisabled
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_OMS_STATE *pOmsState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pOmsState = pInforom->pOmsState;
    if (pOmsState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    *pBDisabled = device->hal.lwswitch_oms_get_device_disable(pOmsState);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_oms_set_device_disable
(
    lwswitch_device *device,
    LwBool bForceDeviceDisable
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_OMS_STATE *pOmsState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pOmsState = pInforom->pOmsState;
    if (pOmsState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    device->hal.lwswitch_oms_set_device_disable(pOmsState, bForceDeviceDisable);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_oms_load
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU8 version = 0;
    LwU8 subversion = 0;
    INFOROM_OMS_STATE *pOmsState = NULL;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = lwswitch_inforom_get_object_version_info(device, "OMS", &version,
                                                    &subversion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO, "no OMS object found, rc:%d\n", status);
        return LWL_SUCCESS;
    }

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(subversion))
    {
        LWSWITCH_PRINT(device, WARN, "OMS v%u.%u not supported\n",
                    version, subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    LWSWITCH_PRINT(device, INFO, "OMS v%u.%u found\n", version, subversion);

    pOmsState = lwswitch_os_malloc(sizeof(INFOROM_OMS_STATE));
    if (pOmsState == NULL)
    {
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(pOmsState, 0, sizeof(INFOROM_OMS_STATE));

    switch (version)
    {
        case 1:
            pOmsState->pFmt = INFOROM_OMS_OBJECT_V1S_FMT;
            pOmsState->pPackedObject = lwswitch_os_malloc(INFOROM_OMS_OBJECT_V1_PACKED_SIZE);
            if (pOmsState->pPackedObject == NULL)
            {
                status = -LWL_NO_MEM;
                goto lwswitch_inforom_oms_version_fail;
            }

            pOmsState->pOms = lwswitch_os_malloc(sizeof(INFOROM_OMS_OBJECT));
            if (pOmsState->pOms == NULL)
            {
                status = -LWL_NO_MEM;
                lwswitch_os_free(pOmsState->pPackedObject);
                goto lwswitch_inforom_oms_version_fail;
            }

            break;

        default:
            LWSWITCH_PRINT(device, WARN, "OMS v%u.%u not supported\n",
                        version, subversion);
            goto lwswitch_inforom_oms_version_fail;
            break;
    }

    status = lwswitch_inforom_load_object(device, pInforom, "OMS",
                                        pOmsState->pFmt,
                                        pOmsState->pPackedObject,
                                        pOmsState->pOms);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load OMS object, rc: %d\n",
                        status);
        goto lwswitch_inforom_oms_load_fail;
    }

    pInforom->pOmsState = pOmsState;

    device->hal.lwswitch_initialize_oms_state(device, pOmsState);

    return LWL_SUCCESS;

lwswitch_inforom_oms_load_fail:
    lwswitch_os_free(pOmsState->pOms);
    lwswitch_os_free(pOmsState->pPackedObject);
lwswitch_inforom_oms_version_fail:
    lwswitch_os_free(pOmsState);

    return status;
}

void
lwswitch_inforom_oms_unload
(
    lwswitch_device *device
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_OMS_STATE *pOmsState;
    LwlStatus status;

    if (pInforom == NULL)
    {
        return;
    }

    pOmsState = pInforom->pOmsState;
    if (pOmsState == NULL)
    {
        return;
    }

    (void)device->hal.lwswitch_read_oob_blacklist_state(device);
    status = device->hal.lwswitch_oms_inforom_flush(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                    "Flushing OMS failed during unload, rc:%d\n", status);
    }

    lwswitch_os_free(pOmsState->pPackedObject);
    lwswitch_os_free(pOmsState->pOms);
    lwswitch_os_free(pOmsState);
    pInforom->pOmsState = NULL;
}
