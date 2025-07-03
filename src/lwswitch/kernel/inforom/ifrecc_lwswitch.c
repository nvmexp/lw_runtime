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
#include "inforom/inforom_lwswitch.h"

LwlStatus
lwswitch_inforom_ecc_load
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU8 version = 0;
    LwU8 subversion = 0;
    INFOROM_ECC_STATE *pEccState = NULL;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = lwswitch_inforom_get_object_version_info(device, "ECC", &version,
                                                    &subversion);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN, "no ECC object found, rc:%d\n", status);
        return LWL_SUCCESS;
    }

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(subversion))
    {
        LWSWITCH_PRINT(device, WARN, "ECC v%u.%u not supported\n",
                    version, subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    LWSWITCH_PRINT(device, INFO, "ECC v%u.%u found\n", version, subversion);

    pEccState = lwswitch_os_malloc(sizeof(INFOROM_ECC_STATE));
    if (pEccState == NULL)
    {
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(pEccState, 0, sizeof(INFOROM_ECC_STATE));

    switch (version)
    {
        case 6:
            pEccState->pFmt = INFOROM_ECC_OBJECT_V6_S0_FMT;
            pEccState->pPackedObject = lwswitch_os_malloc(INFOROM_ECC_OBJECT_V6_S0_PACKED_SIZE);
            if (pEccState->pPackedObject == NULL)
            {
                status = -LWL_NO_MEM;
                goto lwswitch_inforom_ecc_version_fail;
            }

            pEccState->pEcc = lwswitch_os_malloc(sizeof(INFOROM_ECC_OBJECT));
            if (pEccState->pEcc == NULL)
            {
                status = -LWL_NO_MEM;
                lwswitch_os_free(pEccState->pPackedObject);
                goto lwswitch_inforom_ecc_version_fail;
            }

            break;

        default:
            LWSWITCH_PRINT(device, WARN, "ECC v%u.%u not supported\n",
                        version, subversion);
            goto lwswitch_inforom_ecc_version_fail;
            break;
    }

    status = lwswitch_inforom_read_object(device, "ECC", pEccState->pFmt,
                                        pEccState->pPackedObject,
                                        pEccState->pEcc);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to read ECC object, rc:%d\n", status);
        goto lwswitch_inforom_read_fail;
    }

    status = lwswitch_inforom_add_object(pInforom, &pEccState->pEcc->header);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to cache ECC object header, rc:%d\n",
                    status);
        goto lwswitch_inforom_read_fail;
    }

    pInforom->pEccState = pEccState;

    // Update shared surface counts, non-fatal if we encounter a failure
    status = lwswitch_smbpbi_refresh_ecc_counts(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN, "Failed to update ECC counts on SMBPBI "
                       "shared surface rc:%d\n", status);
    }

    return LWL_SUCCESS;

lwswitch_inforom_read_fail:
    lwswitch_os_free(pEccState->pPackedObject);
    lwswitch_os_free(pEccState->pEcc);
lwswitch_inforom_ecc_version_fail:
    lwswitch_os_free(pEccState);

    return status;
}

void
lwswitch_inforom_ecc_unload
(
    lwswitch_device *device
)
{
    INFOROM_ECC_STATE *pEccState;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL)
    {
        return;
    }

    pEccState = pInforom->pEccState;
    if (pEccState == NULL)
    {
        return;
    }

    //
    // Flush the data to InfoROM before unloading the object
    // Lwrrently the driver doesn't support deferred processing and so the
    // error logging path in the interrupt handler cannot defer the flush.
    // This is WAR until the driver adds support for deferred processing
    //
    lwswitch_inforom_ecc_flush(device);

    lwswitch_os_free(pEccState->pPackedObject);
    lwswitch_os_free(pEccState->pEcc);
    lwswitch_os_free(pEccState);
    pInforom->pEccState = NULL;
}

LwlStatus
lwswitch_inforom_ecc_flush
(
    struct lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;
    struct inforom *pInforom = device->pInforom;
    INFOROM_ECC_STATE *pEccState;

    if (pInforom == NULL || pInforom->pEccState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pEccState = pInforom->pEccState;

    if (pEccState->bDirty)
    {
        status = lwswitch_inforom_write_object(device, "ECC",
                                        pEccState->pFmt, pEccState->pEcc,
                                        pEccState->pPackedObject);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to flush ECC object to InfoROM, rc: %d\n", status);
        }
        else
        {
            pEccState->bDirty = LW_FALSE;
        }
    }

    return status;
}

LwlStatus
lwswitch_inforom_ecc_log_err_event
(
    struct lwswitch_device *device,
    INFOROM_LWS_ECC_ERROR_EVENT  *err_event
)
{
    LwlStatus status;
    INFOROM_ECC_STATE *pEccState;
    LwU64 time_ns;
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL || pInforom->pEccState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (err_event == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    pEccState = pInforom->pEccState;

    time_ns = lwswitch_os_get_platform_time();
    err_event->timestamp = (LwU32)(time_ns / LWSWITCH_INTERVAL_1SEC_IN_NS);

    // Scrub the incoming address field if it is invalid
    if (!(err_event->bAddressValid))
    {
        err_event->address = 0;
    }

    // Ilwoke the chip dependent inforom logging routine
    status = device->hal.lwswitch_inforom_ecc_log_error_event(device, pEccState->pEcc,
                                                            err_event);
    if (status == LWL_SUCCESS)
    {
        //
        // If the error was logged successfully, mark the object as dirty to be
        // written on the subsequent flush.
        //
        pEccState->bDirty = LW_TRUE;
    }

    return status;
}

LwlStatus
lwswitch_inforom_ecc_get_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;

    if (pInforom == NULL || pInforom->pEccState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return device->hal.lwswitch_inforom_ecc_get_errors(device, params);
}
