/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOE_LR10_H_
#define _SOE_LR10_H_

#include "lr10.h"

typedef const struct
{

    LwU32 appVersion;
    LwU32 appCodeStartOffset;
    LwU32 appCodeSize;
    LwU32 appCodeImemOffset;
    LwU32 appCodeIsSelwre;
    LwU32 appDataStartOffset;
    LwU32 appDataSize;
    LwU32 appDataDmemOffset;
} SOE_UCODE_APP_INFO_LR10, *PSOE_UCODE_APP_INFO_LR10;

typedef const struct
{

    LwU32 version;
    LwU32 numApps;
    LwU32 codeEntryPoint;
    SOE_UCODE_APP_INFO_LR10 apps[0];
} SOE_UCODE_HDR_INFO_LR10, *PSOE_UCODE_HDR_INFO_LR10;

#define LWSWITCH_SOE_WR32_LR10(_d, _instance, _dev, _reg, _data) \
        LWSWITCH_ENG_WR32_LR10(_d, SOE, , _instance, _dev, _reg, _data)

#define LWSWITCH_SOE_RD32_LR10(_d, _instance, _dev, _reg) \
     LWSWITCH_ENG_RD32_LR10(_d, SOE, _instance, _dev, _reg)

//
// Internal function declarations
//
LwlStatus lwswitch_init_soe_lr10(lwswitch_device *device);
LwlStatus lwswitch_soe_prepare_for_reset_lr10(lwswitch_device *device);
void lwswitch_soe_unregister_events_lr10(lwswitch_device *device);
void lwswitch_therm_soe_callback_lr10(lwswitch_device *device, union RM_FLCN_MSG *pMsg,
         void *pParams, LwU32 seqDesc, LW_STATUS status);
LwlStatus lwswitch_soe_set_ucode_core_lr10(lwswitch_device *device, LwBool bFalcon);
LwlStatus lwswitch_soe_register_event_callbacks_lr10(lwswitch_device *device);
#endif //_SOE_LR10_H_
