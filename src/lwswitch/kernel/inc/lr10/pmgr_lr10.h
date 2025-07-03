/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PMGR_LR10_H_
#define _PMGR_LR10_H_

#include "lr10.h"

void
lwswitch_init_pmgr_lr10
(
    lwswitch_device *device
);

void
lwswitch_init_pmgr_devices_lr10
(
    lwswitch_device *device
);

LwU32
lwswitch_read_physical_id_lr10
(
    lwswitch_device *device
);

LwlStatus
lwswitch_get_rom_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom
);

void
lwswitch_i2c_set_hw_speed_mode_lr10
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 speedMode
);

LwBool
lwswitch_is_i2c_supported_lr10
(
    lwswitch_device *device
);

#endif //_PMGR_LR10_H_
