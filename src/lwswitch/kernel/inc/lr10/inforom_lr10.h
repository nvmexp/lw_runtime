/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _INFOROM_LR10_H_
#define _INFOROM_LR10_H_

LwlStatus
lwswitch_inforom_lwl_get_minion_data_lr10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData
);

LwlStatus
lwswitch_inforom_lwl_set_minion_data_lr10
(
    lwswitch_device     *device,
    void                 *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData,
    LwU32                size,
    LwBool              *bDirty
);

LwlStatus lwswitch_inforom_lwl_log_error_event_lr10
(
    lwswitch_device            *device,
    void                       *pLwlGeneric,
    void                       *pLwlErrorEvent,
    LwBool                     *bDirty
);

LwlStatus lwswitch_inforom_lwl_update_link_correctable_error_info_lr10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pData,
    LwU8 linkId,
    LwU8 lwliptInstance,
    LwU8 localLinkIdx,
    void *pLwlErrorCounts,
    LwBool *bDirty
);

LwlStatus
lwswitch_inforom_lwl_get_max_correctable_error_rate_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
);

LwlStatus
lwswitch_inforom_lwl_get_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
);

LwlStatus
lwswitch_inforom_ecc_log_error_event_lr10
(
    lwswitch_device *device,
    INFOROM_ECC_OBJECT *pEccGeneric,
    INFOROM_LWS_ECC_ERROR_EVENT *err_event
);

void
lwswitch_inforom_ecc_get_total_errors_lr10
(
    lwswitch_device     *device,
    INFOROM_ECC_OBJECT  *pEccGeneric,
    LwU64               *pCorrectedTotal,
    LwU64               *pUncorrectedTotal
);

LwlStatus
lwswitch_inforom_ecc_get_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params
);

void
lwswitch_initialize_oms_state_lr10
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
);

LwBool
lwswitch_oms_get_device_disable_lr10
(
    INFOROM_OMS_STATE *pOmsState
);

void
lwswitch_oms_set_device_disable_lr10
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bForceDeviceDisable
);

LwlStatus
lwswitch_oms_inforom_flush_lr10
(
    struct lwswitch_device *device
);

LwlStatus
lwswitch_bbx_setup_prologue_lr10
(
    lwswitch_device    *device,
    void *pInforomBbxState
);

LwlStatus
lwswitch_bbx_setup_epilogue_lr10
(
    lwswitch_device    *device,
    void *pInforomBbxState
);

LwlStatus
lwswitch_bbx_add_data_time_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

LwlStatus
lwswitch_bbx_add_sxid_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

LwlStatus
lwswitch_bbx_add_temperature_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

void
lwswitch_bbx_set_initial_temperature_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

LwlStatus
lwswitch_inforom_bbx_get_sxid_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
);
#endif //_INFOROM_LR10_H_
