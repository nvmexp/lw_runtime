/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _CLOCK_LR10_H_
#define _CLOCK_LR10_H_

LwlStatus
lwswitch_init_pll_config_lr10
(
    lwswitch_device *device
);

LwlStatus
lwswitch_init_pll_lr10
(
    lwswitch_device *device
);

void
lwswitch_init_hw_counter_lr10
(
    lwswitch_device *device
);

void
lwswitch_hw_counter_shutdown_lr10
(
    lwswitch_device *device
);

LwU64
lwswitch_hw_counter_read_counter_lr10
(
    lwswitch_device *device
);

void
lwswitch_init_clock_gating_lr10
(
    lwswitch_device *device
);

#endif //_CLOCK_LR10_H_
