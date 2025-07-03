/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/inforom_ls10.h"
#include "inforom/ifrstruct.h"

LwlStatus
lwswitch_inforom_lwl_log_error_event_ls10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pLwlErrorEvent,
    LwBool *bDirty
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus lwswitch_inforom_lwl_update_link_correctable_error_info_ls10
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
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_oms_inforom_flush_ls10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

void
lwswitch_initialize_oms_state_ls10
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
)
{
    return;
}

LwBool
lwswitch_oms_get_device_disable_ls10
(
    INFOROM_OMS_STATE *pOmsState
)
{
    return LW_FALSE;
}

void
lwswitch_oms_set_device_disable_ls10
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bForceDeviceDisable
)
{
    return;
}

void
lwswitch_inforom_ecc_get_total_errors_ls10
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
lwswitch_bbx_setup_prologue_ls10
(
    lwswitch_device    *device,
    void               *pInforomBbxState
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_bbx_setup_epilogue_ls10
(
    lwswitch_device    *device,
    void *pInforomBbxState
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_bbx_add_data_time_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_bbx_add_sxid_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

LwlStatus
lwswitch_bbx_add_temperature_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

void
lwswitch_bbx_set_initial_temperature_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return;
}
