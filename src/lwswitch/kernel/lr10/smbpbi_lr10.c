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
#include "lr10/lr10.h"
#include "lr10/smbpbi_lr10.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"
#include "lwswitch/lr10/dev_lwlsaw_ip_addendum.h"


LwlStatus
lwswitch_smbpbi_get_dem_num_messages_lr10
(
    lwswitch_device *device,
    LwU8            *pMsgCount
)
{
    LwU32 reg = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_12);

    *pMsgCount = DRF_VAL(_LWLSAW_SW, _SCRATCH_12, _EVENT_MESSAGE_COUNT, reg);

    return LWL_SUCCESS;
}
