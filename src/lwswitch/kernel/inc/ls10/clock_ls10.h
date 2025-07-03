/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _CLOCK_LS10_H_
#define _CLOCK_LS10_H_

LwlStatus
lwswitch_init_pll_config_ls10
(
    lwswitch_device *device
);

LwlStatus
lwswitch_init_pll_ls10
(
    lwswitch_device *device
);

void
lwswitch_init_clock_gating_ls10
(
    lwswitch_device *device
);

#endif //_CLOCK_LS10_H_
