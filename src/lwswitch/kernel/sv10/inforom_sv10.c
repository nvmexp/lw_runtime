/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"

LwlStatus
lwswitch_inforom_lwl_get_minion_data_sv10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_set_minion_data_sv10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData,
    LwU32                size,
    LwBool              *bDirty
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_log_error_event_sv10
(
    lwswitch_device            *device,
    void                       *pLwlGeneric,
    void                       *pLwlErrorEvent
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus lwswitch_inforom_lwl_update_link_correctable_error_info_sv10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pData,
    LwU8 linkId,
    LwU8 lwliptInstance,
    LwU8 localLinkIdx,
    void *pLwlErrorCounts,
    LwBool *bDirty
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_get_max_correctable_error_rate_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_get_errors_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_ecc_log_error_event_sv10
(
    lwswitch_device *device,
    INFOROM_ECC_OBJECT *pEccGeneric,
    INFOROM_LWS_ECC_ERROR_EVENT *err_event
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}


void
lwswitch_inforom_ecc_get_total_errors_sv10
(
    lwswitch_device    *device,
    INFOROM_ECC_OBJECT *pEccGeneric,
    LwU64              *pCorrectedTotal,
    LwU64              *pUncorrectedTotal
)
{
    return;
}

LwlStatus
lwswitch_inforom_ecc_get_errors_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

void
lwswitch_initialize_oms_state_sv10
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
)
{
    return;
}

LwBool
lwswitch_oms_get_device_disable_sv10
(
    INFOROM_OMS_STATE *pOmsState
)
{
    return LW_FALSE;
}

void
lwswitch_oms_set_device_disable_sv10
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bForceDeviceDisable
)
{
    return;
}

LwlStatus
lwswitch_oms_inforom_flush_sv10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_setup_prologue_sv10
(
    lwswitch_device    *device,
    void *pInforomBbxState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_setup_epilogue_sv10
(
    lwswitch_device    *device,
    void *pInforomBbxState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_data_time_sv10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_sxid_sv10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_temperature_sv10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

void
lwswitch_bbx_set_initial_temperature_sv10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return;
}

LwlStatus
lwswitch_inforom_bbx_get_sxid_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
