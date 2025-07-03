/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _BUS_LR10_H_
#define _BUS_LR10_H_

LwlStatus lwswitch_pex_get_counter_lr10(lwswitch_device *device, LwU32 counterType, LwU32 *pCount);
LwlStatus lwswitch_ctrl_pex_get_lane_counters_lr10(lwswitch_device *device, LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS *pParams);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_pex_clear_counters_lr10(lwswitch_device *device, LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS *pParams);
LwlStatus lwswitch_ctrl_pex_set_eom_lr10(lwswitch_device *device, LWSWITCH_PEX_CTRL_EOM *pParams);
LwlStatus lwswitch_ctrl_pex_get_eom_status_lr10(lwswitch_device *device, LWSWITCH_PEX_GET_EOM_STATUS_PARAMS *pParams);
LwlStatus lwswitch_ctrl_get_uphy_dln_cfg_space_lr10(lwswitch_device *device, LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS *pParams);
LwlStatus lwswitch_ctrl_set_pcie_link_speed_lr10(lwswitch_device *device, LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS *pParams);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#endif //_BUS_LR10_H_
