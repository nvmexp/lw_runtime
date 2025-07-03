/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _THERM_SV10_H_
#define _THERM_SV10_H_

//
// SV10-specific fuse assignments

#define LWSWITCH_FUSE_OPT_TDIODE_CENTER_SV10        LW_FUSE_OPT_RAM_SVOP_PDP
#define LWSWITCH_FUSE_OPT_TDIODE_WEST_SV10       LW_FUSE_OPT_PFG
#define LWSWITCH_FUSE_OPT_TDIODE_EAST_SV10       LW_FUSE_OPT_SPARE_FS


LwlStatus
lwswitch_init_thermal_sv10
(
    lwswitch_device *device
);

LwlStatus
lwswitch_ctrl_therm_read_temperature_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info
);

LwlStatus
lwswitch_ctrl_therm_get_temperature_limit_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS *info
);

void
lwswitch_monitor_thermal_alert_sv10
(
    lwswitch_device *device
);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_force_thermal_slowdown_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p
);

LwlStatus
lwswitch_ctrl_therm_read_voltage_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info
);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#endif //_THERM_SV10_H_
