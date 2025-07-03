/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _INFOROM_LS10_H_
#define _INFOROM_LS10_H_

LwlStatus lwswitch_inforom_lwl_log_error_event_ls10
(
    lwswitch_device            *device,
    void                       *pLwlGeneric,
    void                       *pLwlErrorEvent,
    LwBool                     *bDirty
);

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
);

void
lwswitch_initialize_oms_state_ls10
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
);

LwBool
lwswitch_oms_get_device_disable_ls10
(
    INFOROM_OMS_STATE *pOmsState
);

void
lwswitch_oms_set_device_disable_ls10
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bForceDeviceDisable
);

LwlStatus
lwswitch_oms_inforom_flush_ls10
(
    struct lwswitch_device *device
);

void
lwswitch_inforom_ecc_get_total_errors_ls10
(
    lwswitch_device     *device,
    INFOROM_ECC_OBJECT  *pEccGeneric,
    LwU64               *pCorrectedTotal,
    LwU64               *pUncorrectedTotal
);

LwlStatus
lwswitch_bbx_setup_prologue_ls10
(
    lwswitch_device    *device,
    void *pInforomBbxState
);

LwlStatus
lwswitch_bbx_setup_epilogue_ls10
(
    lwswitch_device    *device,
    void *pInforomBbxState
);

LwlStatus
lwswitch_bbx_add_data_time_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

LwlStatus
lwswitch_bbx_add_sxid_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

LwlStatus
lwswitch_bbx_add_temperature_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

void
lwswitch_bbx_set_initial_temperature_ls10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
);

#endif //_INFOROM_LS10_H_
